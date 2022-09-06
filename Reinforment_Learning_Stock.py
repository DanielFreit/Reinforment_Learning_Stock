import numpy as np

'''For this case study I'm creating a route for automatized operations with Q learning, I'm creating this
manually like in a deep learning scheme, the goal is to create a layout for a inventory/supply deposit,
to automate an agent to understand the best route so the agent can pick the items in the best order possible,
let's say we have an layout like this'''

# PRINT 1

'''Now we have to define the reward and the learning rate for this case, so we set the learning rate to 0.75
and the reward as 0.9'''

gamma = 0.75
alpha = 0.9

#  todo DEFINING THE AMBIENT ------------------------

'''Now we can create a dictionary for every spot that the agent can access in the ambient we're working, we
have 12 shelves so we need to create 12 possible locations, also we're creating the action list and the
array with 0's and 1's, with the 0's being an impossible path, and 1's being a natural and possible path'''

location_to_state = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}

actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]])

# PRINT 2

'''Now we're creating an inverse dictionary so we can use numbers as a key value, instead of letters'''

state_to_location = {state: location for location, state in location_to_state.items()}

'''I'll create a function so we can wrap everything in one place, this part consist in 3 steps, the first step
is the creation of a copy of the ambient, and setting a bigger value for the spot "G" so it can act like a hub for
other locations, the learning where I'm using a Q learning manual method, and the route definition step by step'''


def route(starting_location, ending_location):  # Step 1
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000

    global Q  # Step 2
    Q = np.array(np.zeros([12, 12]))
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state,
                                                                                                      next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

    route = [starting_location]  # Step 3
    next_location = starting_location
    while next_location != ending_location:
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route


'''Basically now the agent have the best route stored in the route variable, this way we can store the values in
a list, and using reward and the learning rate apply the biggest value for a better reward and get to the place
needed with less movements as possible, let's check our Q value to take a look at the scheme the agent find the
best'''

# PRINT 3

'''And check the route setting a hypothetical scenario, let's say we want to go from "E" to "G"'''

print('Route: ')
print(route('E', 'G'))
print(route('G', 'A'))

# PRINT 4

'''Since we're working with products that we don't know the weight and shape, let's create a best route
for out agent, with an intermediary location, so we can control a stop point so the agent can load
in an order that won't make lighter and fragile items be picked before the heavy ones, so we don't crush
the fragile items, also this will give us more control to single and multiple loads per action'''


def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]


print(best_route('E', 'B', 'G'))
print(best_route('G', 'F', 'A'))
print(best_route('A', 'J', 'E'))

# PRINT 5

'''This results can be passed as lists so the agent can understand the order it have to move for a better, faster
and more secure path. We can replace the letters of the alphabet to describe each product area, and with sensors
and typing the name of the products you want from the stock, the agent can pick the items with a simple selection
of which product the agent should pick and the amount'''

# PRINT 6
