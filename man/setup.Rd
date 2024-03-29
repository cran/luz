% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/module.R
\name{setup}
\alias{setup}
\title{Set's up a \code{nn_module} to use with luz}
\usage{
setup(module, loss = NULL, optimizer = NULL, metrics = NULL, backward = NULL)
}
\arguments{
\item{module}{(\code{nn_module}) The \code{nn_module} that you want set up.}

\item{loss}{(\code{function}, optional) An optional function with the signature
\verb{function(input, target)}. It's only requires if your \code{nn_module} doesn't
implement a method called \code{loss}.}

\item{optimizer}{(\code{torch_optimizer}, optional) A function with the signature
\verb{function(parameters, ...)} that is used to initialize an optimizer given
the model parameters.}

\item{metrics}{(\code{list}, optional) A list of metrics to be tracked during
the training procedure. Sometimes, you want some metrics to be evaluated
only during training or validation, in this case you can pass a \code{\link[=luz_metric_set]{luz_metric_set()}}
object to specify mmetrics used in each stage.}

\item{backward}{(\code{function}) A functions that takes the loss scalar values as
it's parameter. It must call \verb{$backward()} or \code{\link[torch:autograd_backward]{torch::autograd_backward()}}.
In general you don't need to set this parameter unless you need to customize
how luz calls the \code{backward()}, for example, if you need to add additional
arguments to the backward call. Note that this becomes a method of the \code{nn_module}
thus can be used by your custom \code{step()} if you override it.}
}
\value{
A luz module that can be trained with \code{\link[=fit]{fit()}}.
}
\description{
The setup function is used to set important attributes and method for \code{nn_modules}
to be used with luz.
}
\details{
It makes sure the module have all the necessary ingredients in order to be fitted.
}
\note{
It also adds a \code{device} active field that can be used to query the current
module \code{device} within methods, with eg \code{self$device}. This is useful when
\code{\link[=ctx]{ctx()}} is not available, eg, when calling methods from outside the \code{luz}
wrappers. Users can override the default by implementing a \code{device} active
method in the input \code{module}.
}
\seealso{
Other training: 
\code{\link{evaluate}()},
\code{\link{fit.luz_module_generator}()},
\code{\link{predict.luz_module_fitted}()}
}
\concept{training}
