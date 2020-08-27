# C19-Update
Contains a script that pulls COVID-19 deaths for the state of Florida from the NY Times repo, fits a time series model to the data, then e-mails a short report with the next day expected number of deaths.

To use this script, you will need to set up a throw away e-mail account. The easiest method (i.e. what I did) is to simply create a gmail account and allow it to use less secure apps (necessary for Python to interface with it). Afterwards, update the Config.py file with the password to your throw-away email and you should be off to the races.

Note: This script could easily be modified to fit models to arbitrary states by wrapping it up in a function and using state as a parameter.
