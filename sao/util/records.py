#
class Records():
#
    def __init__(self,heads):
        self.dim=len(heads)
        self.heads=heads
        self.data=[[None]*self.dim]
#
    def popcol(self,head,data):
#
        index=0
        for h in self.heads:
            if head == h:
                print(index)
                if self.data[-1][index] == None:
                    self.data[-1][index]=data
                else:
                    self.data.append([None]*self.dim)
                    self.data[-1][index]=data
            index=index+1
#
    def getcol(self,head):
        col=[]
        index=0
        for h in self.heads:
            if head == h:
                col_ind=index
            index=index+1
#
        for d in self.data:
            col.append(d[col_ind])
        return col
#
    def getrow(self,date):
        return self.data[date]
