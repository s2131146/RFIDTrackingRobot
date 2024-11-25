class TQueue:
    """独自のキュー（Queueがなんか動かなかったため）"""

    queue: list = []

    def add(self, id, queue=""):
        self.queue.append([self.try_convert_to_int(id), queue])

    def add_all(self, id, queues):
        for queue in queues:
            self.queue.append([self.try_convert_to_int(id), queue])

    def any(self, id_list):
        if not isinstance(id_list, list) or len(id_list) == 0:
            return None

        return any(item[0] in id_list for item in self.queue)

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

    def get_all(self, id):
        if not self.has(id):
            return None

        ret = []
        filtered_data = []
        for item in self.queue:
            if item[0] == id:
                ret.append(item[1])
            else:
                filtered_data.append(item)

        self.queue.clear()
        self.queue = filtered_data

        return ret

    def get_latest(self, id_list):
        if not isinstance(id_list, list) or len(id_list) == 0:
            return None

        ret = None
        filtered_data = []
        for item in self.queue:
            if item[0] in id_list and not isinstance(item[1], bool):
                ret = item[1]
                id_list[:].remove(item[0])
            else:
                filtered_data.append(item)

        self.queue.clear()
        self.queue = filtered_data

        return ret

    def has(self, id):
        if len(self.queue) == 0:
            return False

        return any(q[0] == id for q in self.queue)

    def wait_for(self, id):
        while not self.has(id):
            continue
        self.get(id)

    def empty(self):
        return len(self.queue) == 0

    def get_nowait(self):
        if self.empty():
            return None
        return self.queue.pop(0)

    def try_convert_to_int(self, value):
        try:
            return int(value)
        except ValueError:
            return value
