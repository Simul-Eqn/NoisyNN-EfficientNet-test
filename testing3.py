from utils import get_valid_samplers 
ss = get_valid_samplers(4)
a = list(range(12003))
r1 = ss[0](a)
r2 = ss[1](a)
r3 = ss[2](a)
r4 = ss[3](a)
