#' Create an accelerator
#'
#' @param device_placement (logical) whether the `accelerator` object should
#' handle device placement. Default: `TRUE`
#' @param cpu (logical) whether the training procedure should run on the CPU.
#' @param cuda_index (integer) index of the CUDA device to use if multiple GPUs
#' are available. Default: the result of torch::cuda_current_device().
#'
#' @export
accelerator <- function(device_placement = TRUE, cpu = FALSE, cuda_index = torch::cuda_current_device()) {
  LuzAccelerator$new(device_placement = device_placement, cpu = cpu, cuda_index = cuda_index)
}

LuzAccelerator <- R6::R6Class(
  classname = "LuzAccelerator",
  lock_objects = FALSE,
  public = list(
    initialize = function(device_placement = TRUE, cpu = FALSE, cuda_index = torch::cuda_current_device()) {
      self$device_placement = device_placement
      self$state <- LuzAcceleratorState$new(cpu = cpu, index = cuda_index)
    },
    prepare = function(...) {

      objs <- rlang::list2(...)
      old_parameter_ids <- names(get_parameter_ids(!!!objs))

      results <- lapply(objs, self$prepare_one)
      new_parameter_ids <- get_parameter_ids(!!!results)

      mapping <- setNames(new_parameter_ids, old_parameter_ids)

      if (length(old_parameter_ids) != length(new_parameter_ids))
        rlang::abort(c("Wrong number of parameters in the prepared model.",
                       "Please report an issue in the GitHub repository."))

      switch_parameters(!!!results, .mapping = mapping)

      results
    },
    prepare_one = function(obj) {

      if (torch::is_nn_module(obj))
        return(self$prepare_model(obj))

      if (torch::is_optimizer(obj))
        return(self$prepare_optimizer(obj))

      if (torch::is_dataloader(obj))
        return(self$prepare_dataloader(obj))

      if (is.list(obj))
        return(lapply(obj, self$prepare_one))

      if (is.null(obj))
        return(NULL)

      rlang::abort(glue::glue(
        "Unhandled object with class {class(obj)}\n",
        "Only nn_modules, optimizers and dataloaders are supported."
      ))
    },
    prepare_model = function(model) {
      if (self$device_placement) {
        model <- model$to(device = self$device)
      }
      model
    },
    prepare_optimizer = function(optimizer) {
      optimizer
    },
    prepare_dataloader = function(dataloader) {
      as_device_dataloader(dataloader, self$device)
    }
  ),
  active = list(
    device = function() {
      self$state$device
    }
  )
)

LuzAcceleratorState <- R6::R6Class(
  classname = "LuzAcceleratorState",
  lock_objects = FALSE,
  public = list(
    initialize = function(cpu = FALSE, index = torch::cuda_current_device()) {
      self$device <- private$get_device(cpu, index)
    }
  ),
  private = list(
    get_device = function(cpu, index) {
      if (cpu) return("cpu")

      if (torch::cuda_is_available())
        paste0("cuda:", index)
      else if (torch::backends_mps_is_available())
        "mps"
      else
        "cpu"
    }
  )
)

get_parameter_ids <- function(..., with_parameters) {
  objs <- rlang::list2(...)
  parameters <- list()

  for (obj in objs) {
    if (torch::is_nn_module(obj)) {
      parameters <- append(parameters, obj$parameters)
    }
  }

  names(parameters) <- sapply(parameters, get_param_id)
  parameters
}

switch_parameters <- function(..., .mapping) {
  objs <- rlang::list2(...)
  for (obj in objs) {

    if (torch::is_optimizer(obj)) {
      obj$param_groups <- lapply(
        obj$param_groups,
        function(param_group) {
          param_group$params <- lapply(
            param_group$params,
            function(p) .mapping[[get_param_id(p)]]
          )
          param_group
        }
      )
    }

    # recurse to support getting a list of optimizers
    if (is.list(obj)) {
      switch_parameters(!!!obj, .mapping = .mapping)
    }

  }
  invisible(NULL)
}

get_param_id <- function(p) {
  p$storage()$data_ptr()
}

as_device_dataloader <- function(x, device) {
  x$.device <- device
  class(x) <- c("device_dataloader", class(x))
  x
}

#' @importFrom coro as_iterator
#' @export
as_iterator.device_dataloader <- function(x) {
  g <- NextMethod()
  device <- x$.device
  function() {
    batch <- g()
    to_device(batch, device = device)
  }
}

to_device <- function(batch, device) {
  if (!is.list(batch)) return(batch)
  lapply(batch, function(x) {
    if (inherits(x, "torch_tensor"))
      x$to(device = device, non_blocking = TRUE)
    else if (is.list(x))
      to_device(x, device)
    else
      x
  })
}
