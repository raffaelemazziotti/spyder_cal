import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration_session import SessionSettings, measure_session
import json, sys
s=measure_session(SessionSettings(monitor_name='UI test fixture',num_levels=15,repetitions=3), lambda drive,phase,repetition: .2+160*drive**(2.2+.01*(repetition-2)), lambda *args,**kwargs:None, lambda:None)
s['test_fixture']=True
with open(sys.argv[1],'w',encoding='utf-8') as f: json.dump(s,f)
