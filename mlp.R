mlp <- function(ni=1, nh=5, no=1, lr=0.2)
{
	A <- replicate(nh, rnorm(ni+1, sd=0.01))
	ADelta <- array(1, c(ni+1,nh))
	B <- replicate(no, rnorm(nh+1, sd=0.01))
	mlp <- list(A=A, ADelta=ADelta, B=B, lr=lr)
	class(mlp) <- "mlp"
	mlp
}

train <- function(x, ...) UseMethod("train")

train.mlp <- function(mlp, input, output)
{
	inWithBias <- c(input,1) # concatenate bias input
	outHid <- inWithBias %*% mlp$A # propagate input row through weights between input and hidden layer
	zOutHid <- (exp(outHid) - exp(-outHid))/(exp(outHid) + exp(-outHid)) # apply activation function on hidden output

	netOut <- c(zOutHid,1) %*% mlp$B # concatenate bias input to hidden output and propagates it
	zNetOut <- (exp(netOut) - exp(-netOut))/(exp(netOut) + exp(-netOut)) # calculate network output using activation function

	err = (zNetOut - output) # error
	# BDelta = learningRate * diff(g(x)) * err * hiddenOutputs
	BDelta <- mlp$lr * (1 - zNetOut^2) * err * c(zOutHid,1)

	# Update each weight from inputs to hidden neurons
	for (j in 1:nrow(mlp$A))
		for (k in 1:ncol(mlp$A))
			mlp$ADelta[j,k] <- mlp$lr * (1 - zNetOut^2) * mlp$B[k] * (1 - zOutHid[k]^2) * err * inWithBias[j]

	mlp$A <- mlp$A - mlp$ADelta
	mlp$B <- mlp$B - BDelta
	zNetOut
}

predict.mlp <- function(mlp, input)
{
	inWithBias <- c(input,1)
	outHid <- inWithBias %*% mlp$A
	zOutHid <- (exp(outHid) - exp(-outHid))/(exp(outHid) + exp(-outHid))

	netOut <- c(zOutHid,1) %*% mlp$B
	zNetOut <- (exp(netOut) - exp(-netOut))/(exp(netOut) + exp(-netOut))
    zNetOut
}

neuralnet <- function(data, test, nh=5, lr=0.1, maxSeasons=500, targetErr=0.01)
{
	ni <- dim(data$insts)[2]
	no <- 1
	seasons <- 0
	# A is the matrix of weights between inputs and hidden layer
	A <- replicate(nh, rnorm(ni+1, sd=0.01))
	ADelta <- array(1, c(ni+1,nh))

	# B is the matrix of weights between hidden layer and output
	B <- replicate(no, rnorm(nh+1, sd=0.01))

	converted <- FALSE
	meanErr <- 0
	while(!converted) {
		# presents training instances
		for (i in 1:dim(data$insts)[1])
		{
			inWithBias <- c(data$insts[i,],1) # concatenate bias input
			outHid <- inWithBias %*% A # propagate input row through weights between input and hidden layer
			zOutHid <- (exp(outHid) - exp(-outHid))/(exp(outHid) + exp(-outHid)) # apply activation function on hidden output

			netOut <- c(zOutHid,1) %*% B # concatenate bias input to hidden output and propagates it
			zNetOut <- (exp(netOut) - exp(-netOut))/(exp(netOut) + exp(-netOut)) # calculate network output using activation function

			err = (zNetOut - data$res[i]) # error
			# BDelta = learningRate * diff(g(x)) * err * hiddenOutputs
			BDelta <- lr * (1 - zNetOut^2) * err * c(zOutHid,1)

			# Update each weight from inputs to hidden neurons
			for (j in 1:(ni+1))
				for (k in 1:nh)
					ADelta[j,k] <- lr * (1 - zNetOut^2) * B[k] * (1 - zOutHid[k]^2) * err * inWithBias[j]

			A <- A - ADelta
			B <- B - BDelta
		}

		meanErr <- 0
		# validating
		for (i in 1:dim(data$tInsts)[1] ) {
			outHid <- sapply(c(data$tInsts[i,],1) %*% A, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			netOut <- sapply(c(outHid,1) %*% B, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			err <- (netOut - data$tRes[i])^2/2
			meanErr <- meanErr + err
		}
		t <- meanErr / dim(data$tInsts)[1]
		print (t)
		if (t < targetErr || seasons == maxSeasons )
			converted <- TRUE
	}

	meanTestErr <- 0

	for(i in i:dim(test$insts)[1])
	{
		outHid <- sapply(c(test$insts[i,],1) %*% A, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
		netOut <- sapply(c(outHid,1) %*% B, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
		err <- (netOut - test$res[i])^2/2
		meanTestErr <- meanTestErr + err
	}
	meanTestErr <- meanTestErr / dim(test$insts)[1]
	print ("TestError: ")
	print (meanTestErr)
	list(seasons=seasons, erro=meanErr)

}

