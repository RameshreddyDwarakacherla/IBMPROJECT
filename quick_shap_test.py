#!/usr/bin/env python3
"""Quick test to verify SHAP images are created and can be found"""

import os
import sys

print("🧪 Quick SHAP Test")
print("="*60)

# Change to Frontend directory (where Streamlit runs from)
os.chdir('Multiple-Disease-Prediction-Webapp/Frontend')
print(f"📁 Working directory: {os.getcwd()}")

# Import and run
try:
    sys.path.append('../..')
    from shap_xai_analysis import SHAPAnalyzer
    
    print("\n🔬 Running SHAP analysis for diabetes...")
    analyzer = SHAPAnalyzer()
    analyzer.generate_shap_explanations('diabetes')
    
    # Check if files e
.exit(1)    sys)
xc(_eck.printtraceba    traceback
  import ")
  Error: {e}\n❌ t(f"e:
    prinxception as 
except E       
 .")aboves rorck er Chend. fouges not⚠️  Some ima"\n   print(     :
se  elsis")
  analy try SHAP mlit app andStreatart your : Res("\nNext    print")
     the appow work inould nay shdisplAP "\n✅ The SH print(       )
nd!" fouandated cremages  All i🎉 SUCCESS!t("\n prin    found:
     if all_ 
  = False
   l_found   al       exists:
    not if   )
     _path}"      → {absrint(f"    p    }")
{fatus} "  {st(f  print      bspath(f)
s.path.aabs_path = o        "
s else "❌" if existatus = "✅    ststs(f)
    ath.exi.p exists = os   :
     filesf inue
    for und = Trll_fo    a   ]
    
s.png'
 ence_diabeteependhap_d's,
        betes.png'tance_diahap_impor     'sg',
   abetes.pn_summary_diap     'shs = [
   
    file:") imagesr generateding fo\n📊 Checknt("  pri  xist
