# Overview
This repositroy defines the `stereo` project, and the `rectifi` app. It is the home of `jnickg`'s experiments and investigations into stereoscopy.

The live project can be found at [`rectifi.jnickg.net`](http://rectifi.jnickg.net)

# Algorithms
Algorithm code lives in `./rectifi/algorithm`. The primary algorithm for stereo rectification is [`stereo.py`](./rectifi/algorithm/stereo.py)
# Run Locally
**Note:** For those interested in simply running the stereoscopic rectification algorithm, the easiest way is to run Unit Tests as shown below. Sample input images are provided for testing purposes. To change the files used in the test, replace the files in `./rectifi/test/res/stereo` with files of your choosing (be sure to include calibration-pattern images!).
## Requirements
* Python 3 and `pip`
* Linux systems may require extra apt files. See [`Aptfile`](./Aptfile) for more info.
## Setup
1. `git clone git@github.com:jnickg/jnickg-stereo.git` (or download ZIP file)
2. `cd jnickg-stereo`
3. `pip install -r requirements.txt`
## Running Web Instance Locally
**Note:** In general, see [`Procfile`](./Procfile) (linux) and [`Procfile.windows`](./Procfile.windows) (windows) for commands to run either the web instance, or the worker instance. Below shows only how to run the web instance.

* Windows: `python manage.py runserver 0.0.0.0:5000`  
* Linux: `gunicorn stereo.wsgi --log-file -`
## Running Unit Tests
**Note:** By default, test output files are dumped to the result of `tempfile.gettempdir()`. Look for the actual location in test logs, with the message `MOVED TEST ARTIFACTS TO:`. The output directory can also be easily customized by changing the value of `CONFIG_OUTPUT_DUMP_DIR` in [`test_algorithm.py`](./rectifi/test/test_algorithm.py)
* Windows: `python manage.py test`  
* Linux: `python3 manage.py test`

# Support
Contact `jnickg` with questions, requests, etc.

Track him down by visiting [his website](www.jnickg.net)