module FockStates

export fock_sum, fock_state

"""
FOCKSUM linear combination of harmonic oscillator wave functions

   FOCKSUM(CS,X) where CS and X are vectors, sums a series with
   coefficients CS and evaluates the wave functions at the points X.
   
   FOCKSUM(CS,X) where X is a vector and CS is a matrix, sums the
   series with coefficients CS(:,i) at the point X(i).
   
   The terms of the series are CS(n+1)*F_n(x), where F is the harmonic
   oscillator wave function evaluated by FOCKSTATE.
   
   F_n is a wave function for the harmonic oscillator position quadrature
   (a+a')/sqrt(2), where a is the lowering operator.  A wave function
   for the general quadrature
   
     Y = (a*e^(-it) + a'*e^(it))/sqrt(2)
   
   can be evaluated by propagating CS through t/2*pi of a cycle, i.e.,
   Y = FOCKSUM(CS.*exp(-i*(0:n)*t), XS).  For the momentum quadrature
   Y = i*(a'-a)/sqrt(2), use FOCKSUM(CS.*(-i.^(0:n)), XS).
 
   See also: FOCKSTATE
"""
fock_sum(cs, x::Real) = fock_sum(cs, [x])[]
function fock_sum(cs::AbstractVector{<:Number}, x::AbstractVector{<:Real})

    # For the algorithm, see Clenshaw, "A note on the summation of
    # Chebyshev series", Mathematics of Computation 9, p118 (1955).
    
    # this evaluates a polynomial F_n = H_n/sqrt(2^n*n!), whose
    # recurrence relation can be easily derived from that of the Hermite polynomial H_n.
    
    N = length(cs)-1
    
    B1 = zero(x)				# b_{N+1}
    B2 = zero(x)				# b_{N+2}
    logB = zero(x)			# scale by 2^-logB to prevent overflow
    a = similar(x)				# a_n from F_n recurrence relation
    
    for n = N:-1:0
        a .= -x*sqrt(2/(n+1))	
        b = sqrt((n+1)/(n+2))		# b_{n+1}
        
        B1, B2 = cs[n+1] .- a.*B1 - b*B2, B1
        F = significand.(B1)
        E = exponent.(B1)
        logB += E
        B1, B2 = F, B2./2.0 .^E
    end
    
    π^(-1/4)*exp.(-x.^2/2+log(2)*logB).*B1	# reverse the scaling

end

"""
FOCKSTATE harmonic oscillator wave function

   FOCKSTATE(N,X) evaluates the harmonic oscillator eigenstate F_n(x)
   at the elements of X.
   
   FOCKSTATE(N,X,'matrix'), where X is a vector of length m, returns
   an m-by-(N+1) matrix whose ith column is F_{i-1}(x).
   
   The eigenstates are
   
    F_n(x) = 1/sqrt(2^n*factorial(n))*sqrt(pi))*exp(-x^2/2)*H(n,x),
   
   where H is a Hermite polynomial.  These satisfy the Schrodinger
   equation
   
     -(1/2) F''_n(x) + x^2/2 F_n(x) = (n+1/2) F_n(x).
   
   In terms of the annihilation operator a, the quadrature x is 
   x = (a+a')/sqrt(2).
   
   The Hermite polynomial H_n can be evaluated by the function
   
     H = @(n,x) sqrt(2^n*factorial(n)*sqrt(pi))*exp(-x^2/2) ...
       .*FOCKSTATE(n,x)
   
   The function FOCKSTATE is stable well past the point where the
   Hermite polynomials overflow the double precision floating point
   numbers.
   
   See also: FOCKSUM
"""
fock_state(n::Int, x) = fock_sum([zeros(n); 1.0], x)
fock_state(n, x) = fock_state(convert(Int, n), x)

end # module
