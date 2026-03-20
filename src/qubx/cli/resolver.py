import os
import re
from collections import deque

from qubx import logger

from .release import get_imports


class ModuleResolver:
    """Recursively resolves internal module files and external imports from entry points."""

    def __init__(self, package_root: str, project_root: str, package_name: str):
        """
        Args:
            package_root: Absolute path to the package dir (e.g. .../src/xincubator)
            project_root: Project root for resolve_relative_import() context
            package_name: Top-level package name (e.g. "xincubator")
        """
        self.package_root = os.path.normpath(package_root)
        self.project_root = os.path.normpath(project_root)
        self.package_name = package_name

    def resolve_module_to_file(self, module_parts: list[str]) -> str | None:
        """Resolve a module path to a file on disk.

        Args:
            module_parts: e.g. ["xincubator", "utils", "dataview"]

        Returns:
            Absolute path to the file, or None if external/not found.
        """
        if not module_parts or module_parts[0] != self.package_name:
            return None

        # Strip the package name prefix — the rest is relative to package_root
        relative_parts = module_parts[1:]
        base = os.path.join(self.package_root, *relative_parts) if relative_parts else self.package_root

        # Check: exact .py file
        py_path = base + ".py"
        if os.path.isfile(py_path):
            return py_path

        # Check: exact .pyx file
        pyx_path = base + ".pyx"
        if os.path.isfile(pyx_path):
            return pyx_path

        # Check: directory with __init__.py
        init_path = os.path.join(base, "__init__.py")
        if os.path.isfile(init_path):
            return init_path

        return None

    def collect_internal_files(self, entry_files: list[str]) -> set[str]:
        """BFS from entry files, following all internal imports.

        Args:
            entry_files: List of absolute paths to start from.

        Returns:
            Set of absolute paths of all reachable internal files.
        """
        visited: set[str] = set()
        queue: deque[str] = deque()

        for f in entry_files:
            f = os.path.normpath(f)
            if os.path.isfile(f):
                queue.append(f)

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Parse imports from this file
            try:
                imports = list(get_imports(current, what_to_look=[], project_root=self.project_root))
            except SyntaxError:
                # .pyx files may fail AST parsing — still include the file
                logger.debug(f"SyntaxError parsing {current}, skipping import resolution")
                continue
            except Exception as e:
                logger.warning(f"Failed to parse imports from {current}: {e}")
                continue

            for imp in imports:
                if not imp.module:
                    continue

                # Try resolving the full module path
                resolved = self.resolve_module_to_file(imp.module)
                if resolved and resolved not in visited:
                    queue.append(resolved)

                # For "from pkg.utils import dataview" — check if 'dataview' is a submodule
                if imp.name and isinstance(imp.name, list):
                    for name_part in imp.name:
                        sub_parts = list(imp.module) + [name_part]
                        sub_resolved = self.resolve_module_to_file(sub_parts)
                        if sub_resolved and sub_resolved not in visited:
                            queue.append(sub_resolved)
                elif imp.name and isinstance(imp.name, str):
                    sub_parts = list(imp.module) + [imp.name]
                    sub_resolved = self.resolve_module_to_file(sub_parts)
                    if sub_resolved and sub_resolved not in visited:
                        queue.append(sub_resolved)

        return visited

    def collect_external_imports(self, files: set[str]) -> set[str]:
        """Scan all resolved files for non-internal top-level imports.

        Args:
            files: Set of absolute file paths to scan.

        Returns:
            Set of top-level external module names (e.g. {"numpy", "matplotlib"}).
        """
        external: set[str] = set()

        for fpath in files:
            if fpath.endswith(".pyx"):
                # Regex fallback for .pyx files
                external.update(self._extract_imports_regex(fpath))
                continue

            try:
                for imp in get_imports(fpath, what_to_look=[], project_root=self.project_root):
                    if imp.module and imp.module[0] != self.package_name:
                        external.add(imp.module[0])
            except SyntaxError:
                # Fallback to regex for files with syntax errors
                external.update(self._extract_imports_regex(fpath))
            except Exception as e:
                logger.warning(f"Failed to scan imports from {fpath}: {e}")

        return external

    def _extract_imports_regex(self, fpath: str) -> set[str]:
        """Extract top-level import names using regex (for .pyx or unparseable files)."""
        result: set[str] = set()
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    # import X or import X.Y
                    m = re.match(r'^import\s+([\w.]+)', line)
                    if m:
                        top = m.group(1).split(".")[0]
                        if top != self.package_name:
                            result.add(top)
                        continue
                    # from X import Y or from X.Y import Z
                    m = re.match(r'^from\s+([\w.]+)\s+import\s+', line)
                    if m:
                        top = m.group(1).split(".")[0]
                        if top != self.package_name:
                            result.add(top)
        except Exception as e:
            logger.warning(f"Failed to regex-scan imports from {fpath}: {e}")
        return result

    def ensure_init_files(self, files: set[str]) -> set[str]:
        """For every resolved file, add __init__.py of all parent packages.

        Args:
            files: Set of absolute paths of resolved internal files.

        Returns:
            Augmented set including all parent __init__.py files.
        """
        result = set(files)
        for fpath in files:
            # Walk up from the file's directory to package_root
            current_dir = os.path.dirname(fpath)
            while True:
                # Don't go above or outside package_root
                if not current_dir.startswith(self.package_root):
                    break
                init_file = os.path.join(current_dir, "__init__.py")
                if os.path.isfile(init_file):
                    result.add(init_file)
                if os.path.normpath(current_dir) == self.package_root:
                    break
                current_dir = os.path.dirname(current_dir)
        return result

    def resolve(self, entry_files: list[str]) -> tuple[set[str], set[str]]:
        """Convenience method: resolve entry files to internal files and external imports.

        Runs BFS from entry files, adds parent __init__.py files, then re-scans
        any newly added inits (they may import additional modules). Repeats until
        no new files are discovered.

        Args:
            entry_files: List of absolute paths to strategy entry point files.

        Returns:
            (internal_files, external_imports) tuple.
        """
        internal_files = self.collect_internal_files(entry_files)

        # Iteratively add __init__.py files and scan their imports.
        # __init__.py can re-export siblings (e.g. from .trend_intensity import ...)
        # which must also be resolved and their inits included.
        while True:
            with_inits = self.ensure_init_files(internal_files)
            new_inits = with_inits - internal_files
            if not new_inits:
                break
            # Scan the new init files for additional internal imports
            extra = self.collect_internal_files(list(new_inits))
            internal_files = with_inits | extra
            if extra - with_inits:
                # New files were found — loop to add their __init__.py files too
                continue
            break

        external_imports = self.collect_external_imports(internal_files)
        return internal_files, external_imports
