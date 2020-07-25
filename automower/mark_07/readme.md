# Iterate over multiple lawn sizes, epsilon decay values, learning rate decay values, episode lengths, and discount values.

To run: **iterate_learn_test.py**.

It creates a sqlite database, simple_lawnmower_sjhatfield.sqlite, and writes summary information for each hyperparameter combination.

To see the graphical output of the lawn, in **iterate_learn_test.py**, change `agent_testing.set_debug_render(True)`.

It is also restartable.  It skips over the hyperparameter combinations already explored.  Just restart iterate_learn_test.py to pick up where it left off.  If you need it to explore a hyperparameter combination again, you'll need to delete it from the sqlite database table modelinfo.

To see the likeliest candidates for each lawn size, run this SQL:

`select * from modelinfo where average_run_length < perfect_run_length * 2 order by lawn_size;`

You can execute the SQL while **iterate_learn_test.py** is running.

I have a fairly beefy machine, and it takes ~3 days to run through all of the hyperparameter combinations.
