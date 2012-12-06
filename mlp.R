mlp <- function(ni=1, nh=5, no=1, lr=0.2, activation="sigmoid")
{
	A <- replicate(nh, rnorm(ni+1, sd=0.01))
	ADelta <- array(1, c(ni+1,nh))
	B <- replicate(no, rnorm(nh+1, sd=0.01))
	mlp <- list(A=A, ADelta=ADelta, B=B, lr=lr, activation=activation)
	mlp <- list2env(mlp)
	class(mlp) <- "mlp"
	mlp
}
reset <- function(x, ...) UseMethod("reset")
clone <- function(x, ...) UseMethod("clone")
train <- function(x, ...) UseMethod("train")
predict <- function(x, ...) UseMethod("predict")

sigmoid <- function(x) { 1 / ( 1 + exp(-x) ) }

clone.mlp <- function(m) {
	c <- mlp(nrow(m$A)-1,ncol(m$A),1,m$lr)
	c
}

reset.mlp <- function(m)
{
	A <- replicate(ncol(m$A), rnorm(nrow(m$A), sd=0.01))
	B <- replicate(1, rnorm(ncol(m$A)+1, sd=0.01))
	assign('A', A, envir=m)
	assign('B', B, envir=m)
}

train.mlp <- function(m, input, output)
{
	ADelta <- m$ADelta
	inWithBias <- c(input,1) # concatenate bias input
	outHid <- inWithBias %*% m$A 
	zOutHid <- sigmoid(outHid) 
	
	netOut <- c(zOutHid,1) %*% m$B
	if(m$activation == "sigmoid") zNetOut <- sigmoid(netOut)
	else zNetOut <- tanh(netOut)
	
	err = (zNetOut - output)
	
	if(m$activation == "sigmoid")
		BDelta <- m$lr * (zNetOut*(1-zNetOut)) * err * c(zOutHid,1)
	else
		BDelta <- m$lr * (1 - zNetOut^2) * err * c(zOutHid,1) #tanh
	
	# Update each weight from inputs to hidden neurons
	for (j in 1:nrow(m$A))
		for (k in 1:ncol(m$A))
			if(m$activation == "sigmoid")		
				ADelta[j,k] <- m$lr * (zNetOut*(1-zNetOut)) * m$B[k] * (zOutHid[k]*(1-zOutHid[k])) * err * inWithBias[j]
			else
				ADelta[j,k] <- m$lr * (1 - zNetOut^2) * m$B[k] * (1 - zOutHid[k]^2) * err * inWithBias[j] # 
	
	A <- m$A - ADelta
	B <- m$B - BDelta
	assign('A', A, envir=m)
	assign('B', B, envir=m)
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

