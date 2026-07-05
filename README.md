train_all.py


angle_encoding — takes one row from our data, the 8 numbers, and applies them as rotation angles on the 8 qubits. Each number just tilts its qubit by that angle on the Bloch sphere on the y axis.


learnable_layers — this is the actual brain of the classifier. It goes through 3 rounds. In each round it applies a Rot gate on every qubit using the angles from our params array — these are the numbers that get updated during training(it is stored in our device ram . Then it connects all qubits in a ring using CNOT gates(quantum version of the not gate) so they can share information with each other. Repeating this 3 times lets the circuit learn increasingly complex patterns from the data(its not the three is the right time to do but for the testing I took three didn't test another if needed will test later).


vqc — the full circuit in one function. Calls angle encoding first to load the data, then runs it through the learnable layers, then reads the final orientation of qubit 0 and qubit 1 using PauliZ. Returns two numbers between -1 and +1.


predict_one — takes the two numbers from vqc and converts them to a class. If both are positive it's car clean, positive then negative is car knocking, negative then positive is truck clean, both negative is truck knocking.

 
predict_batch — runs predict_one on every sample in a set and collects all predictions into an array.
accuracy — runs predict_batch and checks how many predictions match the true labels. Returns a percentage.


softmax — converts raw logit numbers into probabilities that sum to 1. Standard helper function used inside the loss.

loss_one — takes one sample, runs it through the circuit, builds 4 logits based on how well the Z0 Z1(are just the qubit values of 0 and 1 from that ) output matches each class's expected sign pattern, converts to probabilities, and returns how wrong the prediction was. Higher loss means more wrong.
batch_loss — runs loss_one on every sample in a batch and returns the average. This is the number we are trying to bring down during training.

compute_gradient — for each of the 72 params it nudges that param up by π/2, runs the batch loss, nudges it down by π/2, runs the batch loss again, and computes the difference. That difference divided by 2 is the exact gradient for that param. This is the parameter shift rule — it tells us which direction to move each param to reduce the loss.

train — the main loop. For each epoch it shuffles the training data, cuts it into batches of 32, computes the gradient for each batch, updates the params by subtracting learning rate times gradient, and prints the loss and accuracy after each epoch. Saves a checkpoint every 10 epochs so we don't lose progress(though it was given in main folder after training I moved them into training files).
evaluate — runs the trained model on the test set and prints the full classification report and confusion matrix so we can see exactly which classes are being confused with which.

train_binary.py

Everything is the same except here we are trying to classify two states car_clean and car_knocking so only two states form the feature data we have we just took those two labels 






update:-

ran the trainings with different parameters.changing learning rate(lr),loss functions,different normalization(0 to pi and 0 to 2 pi) quantum ready has the (0 to pi) and quantum ready 2 has (0 to 2 pi)


