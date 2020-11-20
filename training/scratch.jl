using Profile

function myfunc(a,b)
    return a+b
end

@profile myfunc()

Profile.print(format=:flat)

@time myfunc(3,4)
