import pandas as pd

from qubx.core.basics import CtrlChannel
from qubx.core.helpers import BasicScheduler, _parse_schedule_spec
from qubx.utils.time import timedelta_to_crontab


class TestScheduler:
    def test_parsing_formats(self):
        specs = [
            "",
            "bar: -5Sec",  # 5 sec before subsciption bar closing
            "bar.5m: -5Sec",  # 5 sec before 5Min bar closing
            "bar:  5Sec",  # 5 sec after subsciption bar opened
            "bar:  +5Sec",  # 5 sec after subsciption bar opened
            "time: 23:59:00 @ Sat",  # at 23:59:00 every Saturday
            "time: 23:59 @ Sat",  # at 23:59:00 every Saturday
            "time: 9:30:10",  # every day at 9:30:10
            "* * * * * 45",  # raw cron
            "cron: * * * * * 45",  # raw cron witch implicit type
            "23:59:00 @ Sun",  # at 23:59:00 every Sunday
            "9:30:10",  # every day at 9:30:10
            "9:30 @ Mon-Fri",  # mon, ... fri at 9:30
            "9:30:22 @ Mon,Fri",  # mon and fri at 9:30:22
            "9:30 , 15:45 @ Mon-Fri",  # mon, ... fri at 9:30 and 15:45
            "9:30, 15:45 @ Mon, Fri",  # mon and fri at 9:30 and 15:45
            "-5Sec",  # 5 sec before subsciption bar end
            "5Min -5Sec",  # 5 sec before every 5min interval ends
            "1w -1h",  # 1h before every week ends
            "4month -5hour",  # 5h before the end of every 4 months
            "1week -3hours -10Mins",  # 5h before the end of every 4 months
            "2h30m -5Sec",  # 5 sec before every 2h and 30min interval ends
            "calendar: 23:00 liquidation",
        ]
        res = [
            {},
            {"type": "bar", "spec": "-5Sec", "seconds": "-5"},
            {"type": "bar", "timeframe": "5m", "spec": "-5Sec", "seconds": "-5"},
            {"type": "bar", "spec": "5Sec", "seconds": "5"},
            {"type": "bar", "spec": "+5Sec", "seconds": "+5"},
            {"type": "time", "spec": "23:59:00 @ Sat", "time": "23:59:00 ", "by": "Sat"},
            {"type": "time", "spec": "23:59 @ Sat", "time": "23:59 ", "by": "Sat"},
            {"type": "time", "spec": "9:30:10", "time": "9:30:10"},
            {"spec": "* * * * * 45"},
            {"type": "cron", "spec": "* * * * * 45"},
            {"spec": "23:59:00 @ Sun", "time": "23:59:00 ", "by": "Sun"},
            {"spec": "9:30:10", "time": "9:30:10"},
            {"spec": "9:30 @ Mon-Fri", "time": "9:30 ", "by": "Mon-Fri"},
            {"spec": "9:30:22 @ Mon,Fri", "time": "9:30:22 ", "by": "Mon,Fri"},
            {"spec": "9:30 , 15:45 @ Mon-Fri", "time": "9:30 , 15:45 ", "by": "Mon-Fri"},
            {"spec": "9:30, 15:45 @ Mon, Fri", "time": "9:30, 15:45 ", "by": "Mon, Fri"},
            {"spec": "-5Sec", "seconds": "-5"},
            {"spec": "5Min -5Sec", "minutes": "5", "seconds": "-5"},
            {"spec": "1w -1h", "weeks": "1", "hours": "-1"},
            {"spec": "4month -5hour", "months": "4", "hours": "-5"},
            {"spec": "1week -3hours -10Mins", "weeks": "1", "hours": "-3", "minutes": "-10"},
            {"spec": "2h30m -5Sec", "hours": "2", "minutes": "30", "seconds": "-5"},
            {"type": "calendar", "spec": "23:00 liquidation", "time": "23:00 "},
        ]

        for s, r in zip(specs, res):
            assert _parse_schedule_spec(s) == r

    def test_scheduler(self):
        time_now_fixed = lambda: pd.Timestamp("2024-04-20 12:00:00", tz="UTC").as_unit("ns").asm8.item()  # noqa: E731

        bs = BasicScheduler(CtrlChannel("test"), time_now_fixed)

        # schedule every 10 sec
        bs.schedule_event("* * * * * */10", "TEST")
        assert "* * * * * */10" == bs.get_schedule_for_event("TEST")

        assert bs.get_event_last_time("TEST") == pd.Timestamp("2024-04-20 11:59:50")
        assert bs.get_event_next_time("TEST") == pd.Timestamp("2024-04-20 12:00:10")

    def test_scheduler_from_timedelta(self):
        time_now_fixed = lambda: pd.Timestamp("2024-04-20 12:01:00", tz="UTC").as_unit("ns").asm8.item()  # noqa: E731

        bs = BasicScheduler(CtrlChannel("test"), time_now_fixed)

        sched = timedelta_to_crontab(pd.Timedelta("1d23h55Min"))
        bs.schedule_event(sched, "TEST1")

        sched = timedelta_to_crontab(pd.Timedelta("1h"))
        bs.schedule_event(sched, "TEST2")

        sched = timedelta_to_crontab(pd.Timedelta("15Min1Sec"))
        bs.schedule_event(sched, "TEST3")

        assert bs.get_event_next_time("TEST1") == pd.Timestamp("2024-04-20 23:55:00")
        assert bs.get_event_next_time("TEST2") == pd.Timestamp("2024-04-20 13:00:00")
        assert bs.get_event_next_time("TEST3") == pd.Timestamp("2024-04-20 12:15:01")

    def test_scheduler_run(self):
        time_now = lambda: pd.Timestamp("now", tz="UTC").as_unit("ns").asm8.item()  # noqa: E731
        bs = BasicScheduler(c := CtrlChannel("test"), time_now)

        # - schedule event every 1 and 2 sec
        bs.schedule_event("* * * * * */1", "test-1")
        bs.schedule_event("* * * * * */2", "test-2")

        # - finishing event
        t = pd.Timestamp(time_now(), unit="ns") + pd.Timedelta("7s")
        bs.schedule_event(f"{t.minute} {t.hour} {t.day} {t.month} * {t.second}", "test-3")
        bs.run()

        t1, t2 = 0, 0
        while c.control.is_set():
            s, event, data, _ = c.receive()
            print(event, data)
            if event == "test-1":
                t1 += 1
            if event == "test-2":
                t2 += 1
            if event == "test-3":
                c.control.clear()
        assert t1 >= 6
        assert t2 >= 3

    def test_scheduler_test(self):
        """
        Test scheduler in pseudo simulation context when we don't need any references to actual time
        """
        from queue import Empty

        class TesterScheduler(BasicScheduler):
            def run(self):
                self._is_started = True

        class PseudoBacktester:
            def __init__(self) -> None:
                self.chan = CtrlChannel("test")
                self.c_time = pd.Timestamp("2024-04-20 10:00")
                self.scheduler = TesterScheduler(self.chan, self.time_now)
                self.scheduler.run()

                # - wakeup once per second
                self.scheduler.schedule_event("* * * * * */1", "test-1")

            def time_now(self):
                return self.c_time.as_unit("ns").asm8.item()

            def run_test(self):
                for i in range(20):
                    self.c_time += pd.Timedelta("0.5s")
                    self.scheduler.check_and_run_tasks()

                # - read the queue back
                n = 0
                try:
                    while True:
                        print(self.chan._queue.get(block=False))
                        n += 1
                except Empty as e:
                    pass
                print(f"DONE: {n}")
                return n

        tester = PseudoBacktester()
        assert tester.run_test() == 10

    def test_unschedule_event(self):
        """Test that unscheduling an event removes it properly"""
        time_now_fixed = lambda: pd.Timestamp("2024-04-20 12:00:00", tz="UTC").as_unit("ns").asm8.item()  # noqa: E731

        bs = BasicScheduler(CtrlChannel("test"), time_now_fixed)

        # Schedule event
        bs.schedule_event("* * * * * */10", "TEST")
        assert bs.get_schedule_for_event("TEST") == "* * * * * */10"

        # Unschedule event
        assert bs.unschedule_event("TEST") is True
        assert bs.get_schedule_for_event("TEST") is None

        # Try to unschedule non-existent event
        assert bs.unschedule_event("NONEXISTENT") is False

    def test_reschedule_event(self):
        """Test that rescheduling an event removes the old schedule"""
        time_now_fixed = lambda: pd.Timestamp("2024-04-20 12:00:00", tz="UTC").as_unit("ns").asm8.item()  # noqa: E731

        bs = BasicScheduler(CtrlChannel("test"), time_now_fixed)

        # Schedule event at 10 second intervals
        bs.schedule_event("* * * * * */10", "TEST")
        assert bs.get_event_next_time("TEST") == pd.Timestamp("2024-04-20 12:00:10")

        # Reschedule to 20 second intervals
        bs.schedule_event("* * * * * */20", "TEST")
        assert bs.get_event_next_time("TEST") == pd.Timestamp("2024-04-20 12:00:20")
