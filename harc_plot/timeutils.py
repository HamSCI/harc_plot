import datetime

def strip_time(dt):
    return datetime.datetime(dt.year,dt.month,dt.day)

def daterange(sTime,eTime):
    """ Get every date in the range start_date -> end_date (inc) """
    start_date  = strip_time(sTime)
    end_date    = strip_time(eTime)
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + datetime.timedelta(n)

def dt64_to_ut_hours(dt64):
    """ Get UT hours in a decimal from datetime64 """
    sec = dt64.astype("M8[s]")
    return sec.astype(int) % (24 * 3600) / 3600

def ut_hours_to_slt(ut_hours, long):
    """ Get SLT at a longitude from UT hours """
    return (ut_hrs + long/15.) % 24
