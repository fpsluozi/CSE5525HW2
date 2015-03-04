import math

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = ['']
    epsilon = 0.00000000000000000001
 
    # Initialize base cases (t == 0)
    # Actually it will already be <s>

    max_v = -1.0 / epsilon / 1000
    for y in states:
        temp_emit = emit_p[y].prob(obs[0])
        if temp_emit < epsilon:
            temp_emit = epsilon
        V[0][y] = math.log(start_p[y]) + math.log(temp_emit)
        if max_v < V[0][y]: 
            max_v = V[0][y]          
            path[0] = y
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
         
        V.append({})
        path.append('')
        max_v = -1.0 / epsilon
        for y in states:
            
            if y == "<s>":
                V[t][y] = max_v
            else:
                temp = list()
                for y0 in states: 
                    temp_trans = trans_p[y0].prob(y)
                    if temp_trans < epsilon:
                        temp_trans = epsilon
                    temp_prev_v = V[t-1][y0]
                    if temp_prev_v < epsilon:
                        temp_prev_v = epsilon
                    temp.append((math.log(temp_prev_v) + math.log(temp_trans)))

                temp_emit = emit_p[y].prob(obs[t])
                if temp_emit < epsilon:
                    temp_emit = epsilon
                prob = math.log(temp_emit) + max(temp)
                V[t][y] = prob
                if max_v < prob:              
                    max_v = prob
                    path[t] = y

    # Outputting the Viterbi path    
    return path