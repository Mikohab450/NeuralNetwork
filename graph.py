import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model_name ="model"

def create_acc_loss_graph(model_name):
    contents=open("model(120,50,20).log","r").read().split('\n')
    times=[]
    list_acc=[]
    list_loss=[]
    list_val_acc=[]
    list_val_loss=[]
    epochs=[]
    for c in contents:
        if model_name in c:
            name, timestamp,epoch,acc,loss,val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            list_acc.append(float(acc))
            list_loss.append(float(loss))
            list_val_acc.append(float(val_acc))
            list_val_loss.append(float(val_loss))
            epochs.append(float(epoch))

    fig=plt.figure()
    ax1=plt.subplot2grid((2,1),(0,0))
    ax2=plt.subplot2grid((2,1),(1,0),sharex=ax1)
    ax1.plot(times,list_acc,label="acc")
    ax1.plot(times,list_val_acc,label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times,list_loss,label="loss")
    ax2.plot(times,list_val_loss,label="vall_loss")
    ax2.legend(loc=2)

    plt.show()

create_acc_loss_graph(model_name)