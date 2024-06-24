class TQueue:
    queue: list = []

    def add(self, id, queue):
        self.queue.append([id, queue])

    def get(self, id):
        if not self.has(id):
            return None   
         
        ret = ""
        found = False
        filtered_data = []
        for item in self.queue:
            if item[0] == id and not found:
                ret = item[1]
                found = True
            else:
                filtered_data.append(item)
        
        self.queue.clear()
        self.queue = filtered_data

        return ret
    
    def has(self, id):
        if len(self.queue) == 0:
            return False
        
        return any(q[0] == id for q in self.queue)