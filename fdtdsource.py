def fdtdsource(ss,freq,omega,dt,nmax,etaz):
    #fdtdsource(ss,freq,omega,dt,nmax,etaz)
    # source = function source
    # ss = true for sinusoidal
    if not ss:
        rtau = 1 / (5e9 * math.pi)
        tau = rtau / dt
        delay = 3 * tau
        mag = 1.148
    source = numpy.zeros(nmax)
    if ss:
        source = math.cos(2 * math.pi * freq * [x for x in range(nmax)] * dt)  #steady state
    else:
        for n in range(7.0 * tau):
            source[n] = etaz * mag * math.sin(omega * (n - delay) * dt) * exp(-((n - delay)^2 / tau^2))
    return source
'''
if ss == true
    source = cos(2*pi*freq*(1:nmax)*dt); % steady state
else
   for n=1:7.0*tau
       %  source(n)=sin(omega*(n-delay)*dt)*exp(-((n-delay)^2/tau^2)); 
        source(n)=etaz*mag*sin(omega*(n-delay)*dt)*exp(-((n-delay)^2/tau^2));
    end
end
'''