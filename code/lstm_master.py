from multiprocessing import Process
import numpy
import time

import lstm
import channel


class LSTMLieutenant(channel.Lieutenant):
    def __init__(self, max_mb, ydim, patience):
        channel.Lieutenant.__init__(self)
        self.max_mb = max_mb
        self.ydim = int(ydim)
        self.patience = patience

        self.uidx = 0
        self.eidx = 0
        self.history_errs = []
        self.bad_counter = 0

        self.stop = False
        self.start_time = None

    def init_mb(self, dataset, n_words, maxlen, batch_size, max_epochs):
        load_data, self.prepare_data = lstm.get_dataset(dataset)

        self.batch_size = batch_size

        print "Loading data"
        self.train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                            maxlen=maxlen)

        self.ydim = int(numpy.max(self.train[1]) + 1)

        print "%d train examples" % len(self.train[0])

        self.max_mb = ((len(self.train[0]) * max_epochs) // batch_size) + 1

    def start_mb_server(self, port):
        self.p = Process(target=self.do_mb, args=(port,))
        self.p.start()

    def do_mb(self, port):
        self.init_data(port=port)
        while True:
            kf = lstm.get_minibatches_idx(len(self.train[0]), self.batch_size, shuffle=True)
            for _, train_index in kf:
                # Select the random examples for this minibatch
                y = [self.train[1][t] for t in train_index]
                x = [self.train[0][t] for t in train_index]

                x, mask, y = self.prepare_data(x, y)

                self.send_mb([x, mask, y])


    def handle_control(self, req):
        if req == 'next':
            if self.start_time is None:
                self.start_time = time.time()
            if self.stop:
                return 'stop'
            return 'train'
        if req == 'ydim':
            return self.ydim
        if isinstance(req, dict):
            if 'done' in req:
                self.uidx += req['done']
                if self.uidx > self.max_mb:
                    self.stop = True
                    self.stop_time = time.time()
                    print "Training time %fs" % (self.stop_time - self.start_time,)
                    return 'stop'
            if 'valid_err' in req:
                valid_err = req['valid_err']
                test_err = req['test_err']
                self.history_errs.append([valid_err, test_err])
                harr = numpy.array(self.history_errs)[:, 0]
                if valid_err <= harr.min():
                    self.bad_counter = 0
                    return 'best'
                if (len(self.history_errs) > self.patience and
                        valid_err >= harr[:-self.patience].min()):
                    self.bad_counter += 1
                    if self.bad_counter > self.patience:
                        self.stop_time = time.time()
                        print "Training time %fs" % (self.stop_time - self.start_time,)
                        self.stop = True
                        return 'stop'

def lstm_control(dataset='imdb',
                 patience=10,
                 test_size=-1,
                 n_words=10000,
                 maxlen=100,
                 dispFreq=10,
                 max_epochs=5000,
                 validFreq=370,
                 saveFreq=1110,
                 batch_size=64,
                 valid_batch_size=64,
                 saveto=None,
                 ):

    l = LSTMLieutenant(max_mb=0, ydim=0, patience=patience)

    l.init_mb(dataset, n_words, maxlen, batch_size, max_epochs)
    l.start_mb_server(5566)

    l.init_control(port=5567)
    print "Lieutenant is ready"
    l.serve()

if __name__ == '__main__':
    lstm_control(max_epochs=10)
