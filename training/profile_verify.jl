using Profile

include("profile.jl")

@profile math_prob(3,4)

Profile.print()
