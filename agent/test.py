from .tools import calculate_distance


result = calculate_distance.invoke(
    {
        "origin": "raipur ahmedabad ?",
        "destination":"lg hosptial maninagar, ahemdabad"
    }
)

print(result)