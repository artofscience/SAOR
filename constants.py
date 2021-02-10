## Constants file

## Complete lists to choose from

# All the implemented problems
PROBLEM_LIST = ['Li2015Fig4', 'RosenCubic', 'MishraBird', 'Townsend', 'Simionescu']

# All the implemented approximations
APPROX_LIST = ['Linear', 'MMA', 'MixedTemplate']

# All the implemented convergence criteria
CONVERGENCE_CRITERIA_LIST = ['KKT', 'Feasibility', 'ObjectiveChange', 'VariableChange', 'AllTogether']


## Parameters that are going to be used in the main()

MAX_ITE_OPT = 100                      # Maximum number of optimization iterations
TOLERANCE = 1e-4                       # The tolerance for the convergence criterion
MOVE_LIMIT = 0.1                       # % of (xmax - xmin) that a variable is allowed to change
