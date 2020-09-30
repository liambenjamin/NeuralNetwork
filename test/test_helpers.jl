#Author: Liam Johnston
#Date: June 4, 2020
#Description: Test functions in Helpers


include("../src/helpers.jl")

using LinearAlgebra, Colors, Main.Helpers


printstyled("\tTesting getInputIndices():", color=:yellow)
    #(input dim + hidden dim, hidden_dim, seq_length )
    Helpers.getInputIndices(6, 2, 2) == ([1, 3], [2,4]) ?
                                printstyled("\t\tPASSED", color=:green) :
                                printsytled("\t\tFAILED", color=:red)

    Helpers.getInputIndices(13, 4, 3) == ([1,4,7], [3,6,9]) ?
                                printstyled("\t\tPASSED", color=:green) :
                                printsytled("\t\tFAILED", color=:red)

    Helpers.getInputIndices(19, 3, 4) == ([1,5,9,13], [4,8,12,16]) ?
                                printstyled("\t\tPASSED\n\n", color=:green) :
                                printsytled("\t\tFAILED\n\n", color=:red)

printstyled("\tTesting getStackedStateIndices():", color=:yellow)
    #(input_dim, hid_dim, seq_length)

    Helpers.getStackedStateIndices(6, 2, 2) == 5:10 ?
                                printstyled("\t\tPASSED", color=:green) :
                                printsytled("\t\tFAILED", color=:red)

    Helpers.getStackedStateIndices(13, 4, 3) == 10:25 ?
                                printstyled("\t\tPASSED", color=:green) :
                                printsytled("\t\tFAILED", color=:red)

    Helpers.getStackedStateIndices(19, 3, 4) == 17:31 ?
                                printstyled("\t\tPASSED\n\n", color=:green) :
                                printsytled("\t\tFAILED\n\n", color=:red)
