#!/bin/bash
# Linux-only: navigates TORCS menus via xte keypresses.
# On Windows, web_app/backend/simulator.py uses PowerShell + PostMessage instead.
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Up'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
