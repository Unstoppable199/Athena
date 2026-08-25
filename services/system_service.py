"""
System capability.

Deterministic system information that requires no external
call or reasoning (date, time, timezone conversions).
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class SystemService:

    def datetime_now(self, timezone: str = None):

        try:

            # The planner says "system" or "local" when it means this
            # machine's own clock, which is not a zone name and was
            # rejected as unknown - so "what year is it" came back as
            # the words "Unknown timezone: system". Treated as no zone
            # at all, which is what it means.
            if timezone and str(timezone).strip().lower() in {
                "system", "local", "localtime", "here", "default", "none",
            }:
                timezone = None

            if timezone:

                try:
                    tz = ZoneInfo(timezone)

                except ZoneInfoNotFoundError:

                    return {
                        "success": False,
                        "error": f"Unknown timezone: {timezone}"
                    }

                now = datetime.now(tz)

            else:

                now = datetime.now()

            return {
                "success": True,
                "data": {
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "day_of_week": now.strftime("%A"),
                    "timezone": timezone or "local",
                    "iso": now.isoformat()
                }
            }

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }
            
    def duration_since(self, start_date: str, end_date: str = None):
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date) if end_date else datetime.now()
            delta = end - start
            years = delta.days // 365
            months = (delta.days % 365) // 30
            return {
                "success": True,
                "data": {
                    "start": start_date,
                    "end": end_date or "ongoing",
                    "years": years,
                    "months": months,
                    "total_days": delta.days
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}