#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:05:27 2019

@author: sumyiren and nathan

"""

from testdrqnUser import testDrqn
import tkinter as tk

#test = testDrqn()
#test.restart()
#a = test.stepAction(0)
#print(a)

backend = testDrqn()
backend.restart()


ourcounter = 0
#theircounter = 0
#def counter_label(label):
#  def count():
#    global counter
#    counter += 1
#    label.config(text=str(counter))
#    label.after(1000, count)
#  count()

#def ourcounter_label(label):
#    def ourcount():
#        print('HERE')
#        global ourcounter
#        ourcounter += 1
#        label.config(text=str(ourcounter))
##        label.after(1000, ourcount)
#    ourcount()
        
def up_ourcounter():
    global ourcounter
    ourcounter += 1
    global label
    global label1
    a = backend.stepAction(2)
    print(a)
    print(a[0][0])
    label.config(text=str(a[0][1]))
    label1.config(text=str(a[0][0])) 
#    global ourcounter
#    ourcounter += 1
#    label.config(text=str(ourcounter))
def down_ourcounter():
    global ourcounter
    ourcounter -= 1
    global label
    global label1
    b = backend.stepAction(0)
    print(b)
    print(b[0][0])
    label.config(text=str(b[0][1]))
    label1.config(text=str(b[0][0])) 
 
def same_ourcounter():
    global ourcounter
    ourcounter = ourcounter
    global label
    global label1
    c = backend.stepAction(1)
    print(c)
    print(c[0][0])
    label.config(text=str(c[0][1]))    
    label1.config(text=str(c[0][0])) 
root = tk.Tk()
root.title("Counting Seconds")
root.geometry("500x500") #You want the size of the app to be 500x500
root.resizable(0, 0) #Don't allow resizing in the x or y direction

back = tk.Frame(master=root,bg='white')
back.pack_propagate(0) #Don't allow the widgets inside to determine the frame's width / height
back.pack(fill=tk.BOTH, expand=1) #Expand the frame to fill the root window

label = tk.Label(root)
label.pack()
label1 = tk.Label(root)
label1.pack()
#ourcounter_label(label)
#counter_label(label1)
button1 = tk.Button(root, text='Up', width=30, command=up_ourcounter)
button2 = tk.Button(root, text='Down', width=35, command=down_ourcounter)
button3 = tk.Button(root, text='Same', width=40, command=same_ourcounter)
label.place(height=50, width=50, x= 80, y = 250)
label1.place(height=50, width=50, x= 380, y = 250)
button1.pack()
button1.place(height=80, width=80, x = 100, y = 100)
button2.pack()
button2.place(height=80, width=80, x = 300, y = 100)
button3.pack()
button3.place(height=80, width=80, x = 200, y = 100)
root.mainloop()
