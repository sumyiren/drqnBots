#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, redirect, url_for, request
from testdrqnUser import testDrqn

app = Flask(__name__)

@app.route('/start',methods = ['POST'])
def drqn():
   if request.method == 'POST':
      user = request.form['nm']
      
      return redirect(url_for('success',name = user))

if __name__ == '__main__':
   app.run()