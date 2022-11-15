#.....draw letter M with python turtle........!
import turtle
import time
t=turtle.Turtle()
t.penup()
#draw straight line
t.goto(-150,50) #centering in the screen
t.pendown()
t.pensize(10)
t.pencolor("red")
 
t.right(90)
t.forward(150)
 
t.goto(-150,50)
t.goto(-100,-20)
t.goto(-55,50)
t.goto(-55,-100)

#t.right(90)
#t.forward(150)

t.goto(100,-100)
t.goto(100,50)
t.goto(150,-20)
t.goto(195,50)
t.goto(195,-100)
time.sleep(5)
