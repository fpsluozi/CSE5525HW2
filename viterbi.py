import numpy
class Decoder(object):
    #create object with initial prob, trans prob and emission prob
    def __init__(self, initialProb, transProb, obsProb):
        self.N = initialProb.shape[0]
        self.initialProb = initialProb
        self.transProb = transProb
        self.obsProb = obsProb
        assert self.initialProb.shape == (self.N, 1)
        assert self.transProb.shape == (self.N, self.N)
        assert self.obsProb.shape[0] == self.N
 
    def Obs(self, obs):
        return self.obsProb[:, obs, None]
    def Viterbi(self, obs):
        #initialization
        viterbi = numpy.zeros((self.N, len(obs)))
        backpt = numpy.ones((self.N, len(obs)), 'int32') * -1
        viterbi[:, 0] = numpy.squeeze(self.initialProb * self.Obs(obs[0]))
        #recursion
        for t in xrange(1, len(obs)):
            viterbi[:, t] = (viterbi[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            backpt[:, t] = (numpy.tile(viterbi[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        tokens = [viterbi[:, -1].argmax()]
        for i in xrange(len(obs)-1, 0, -1):
        tokens.append(backpt[tokens[-1], i])
        return tokens[::-1]