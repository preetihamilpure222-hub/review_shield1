"""Launcher so the project runs from the project root:
   local :  python app.py            ->  http://127.0.0.1:5000
   Render:  gunicorn app:app ...     (also works: gunicorn --chdir apps app:app)"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APPS = os.path.join(ROOT, "apps")
sys.path.insert(0, APPS)
_spec = importlib.util.spec_from_file_location("reviewshield_app", os.path.join(APPS, "app.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["reviewshield_app"] = _mod
_spec.loader.exec_module(_mod)
app = _mod.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    online = "PORT" in os.environ
    app.run(host="0.0.0.0" if online else "127.0.0.1", port=port, debug=not online)
