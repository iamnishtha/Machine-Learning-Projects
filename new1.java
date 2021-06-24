#import java.util.*;
class Stack
{
    int max=1000;
    int top;
    int a[]=new int[max];

    boolean isEmpty()
    {
        return (top<0);
    }
    Stack ()
    {
        top=-1;
    }
    
    boolean push(int x)
    {
        if (top>=(max-1))
        {
            System.out.println("Stack Overflow");
            return false;
        }
        else
        {
            a[++top]=x;
            System.out.println(x+" pushed into the Stack");
            return true;
        }
    }
    int pop()
    {
        if (top<0){
            System.out.println("Stack Underflow");
            return 0;
        }
        else {
            int x=a[top--];
            return x
        }
    }
    public static void main(String args[])
    {
        Stack a =new Stack();
        a.push(10);
        a.push(50);
        a.push(60);
        System.out.println(s.pop+" Popped from stack");
    }


