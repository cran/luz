## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(luz)
library(torch)

## ---- eval = FALSE------------------------------------------------------------
#  net <- nn_module(
#    "Net",
#    initialize = function(num_class) {
#      self$conv1 <- nn_conv2d(1, 32, 3, 1)
#      self$conv2 <- nn_conv2d(32, 64, 3, 1)
#      self$dropout1 <- nn_dropout2d(0.25)
#      self$dropout2 <- nn_dropout2d(0.5)
#      self$fc1 <- nn_linear(9216, 128)
#      self$fc2 <- nn_linear(128, num_class)
#    },
#    forward = function(x) {
#      x <- self$conv1(x)
#      x <- nnf_relu(x)
#      x <- self$conv2(x)
#      x <- nnf_relu(x)
#      x <- nnf_max_pool2d(x, 2)
#      x <- self$dropout1(x)
#      x <- torch_flatten(x, start_dim = 2)
#      x <- self$fc1(x)
#      x <- nnf_relu(x)
#      x <- self$dropout2(x)
#      x <- self$fc2(x)
#      x
#    }
#  )

## ---- eval = FALSE------------------------------------------------------------
#  fitted <- net %>%
#    setup(
#      loss = nn_cross_entropy_loss(),
#      optimizer = optim_adam,
#      metrics = list(
#        luz_metric_accuracy
#      )
#    ) %>%
#    set_hprams(num_class = 10) %>%
#    set_opt_hparams(lr = 0.003) %>%
#    fit(train_dl, epochs = 10, valid_data = test_dl)

## ---- eval = FALSE------------------------------------------------------------
#  predictions <- predict(fitted, test_dl)

## ---- eval = FALSE------------------------------------------------------------
#  # -> Initialize objects: model, optimizers.
#  # -> Select fitting device.
#  # -> Move data, model, optimizers to the selected device.
#  # -> Start training
#  for (epoch in 1:epochs) {
#    # -> Training procedure
#    for (batch in train_dl) {
#      # -> Calculate model `forward` method.
#      # -> Calulate the loss
#      # -> Update weights
#      # -> Update metrics and tracking loss
#    }
#    # -> Validation procedure
#    for (batch in valid_dl) {
#      # -> Calculate model `forward` method.
#      # -> Calulate the loss
#      # -> Update metrics and tracking loss
#    }
#  }
#  # -> End training

## ---- eval=FALSE--------------------------------------------------------------
#  fitted <- net %>%
#    setup(
#      ...
#      metrics = list(
#        luz_metric_accuracy
#      )
#    ) %>%
#    fit(...)

## ---- eval = FALSE------------------------------------------------------------
#  luz_metric_accuracy <- luz_metric(
#    # An abbreviation to be shown in progress bars, or
#    # when printing progress
#    abbrev = "Acc",
#    # Initial setup for the metric. Metrics are initialized
#    # every epoch, for both training and validation
#    initialize = function() {
#      self$correct <- 0
#      self$total <- 0
#    },
#    # Run at every training or validation step and updates
#    # the internal state. The update function takes `preds`
#    # and `target` as parameters.
#    update = function(preds, target) {
#      pred <- torch::torch_argmax(preds, dim = 2)
#      self$correct <- self$correct + (pred == target)$
#        to(dtype = torch::torch_float())$
#        sum()$
#        item()
#      self$total <- self$total + pred$numel()
#    },
#    # Use the internal state to query the metric value
#    compute = function() {
#      self$correct/self$total
#    }
#  )

## ----include=FALSE, eval = torch::torch_is_installed()------------------------
library(luz)
torch::torch_manual_seed(1)
get_model <- function() {
  torch::nn_module(
    initialize = function(input_size, output_size) {
      self$fc <- torch::nn_linear(prod(input_size), prod(output_size))
      self$output_size <- output_size
    },
    forward = function(x) {
      out <- x %>%
        torch::torch_flatten(start_dim = 2) %>%
        self$fc()
      out$view(c(x$shape[1], self$output_size))
    }
  )
}

model <- get_model()
model <- model %>%
  setup(
    loss = torch::nn_mse_loss(),
    optimizer = torch::optim_adam,
    metrics = list(
      luz_metric_mae(),
      luz_metric_mse(),
      luz_metric_rmse()
    )
  ) %>%
  set_hparams(input_size = 10, output_size = 1) %>%
  set_opt_hparams(lr = 0.001)

x <- list(torch::torch_randn(100,10), torch::torch_randn(100, 1))

fitted <- model %>% fit(
  x,
  epochs = 1,
  verbose = FALSE,
  dataloader_options = list(batch_size = 2, shuffle = FALSE)
)

evaluation <- fitted %>% evaluate(data = x)

## ---- eval = FALSE------------------------------------------------------------
#  evaluation <- fitted %>% evaluate(data = valid_dl)
#  metrics <- get_metrics(evaluation)
#  print(evaluation)

## ----echo=FALSE, eval=torch::torch_is_installed()-----------------------------
options(cli.unicode = FALSE)
metrics <- get_metrics(evaluation)
print(evaluation)

## ---- eval = FALSE------------------------------------------------------------
#  print_callback <- luz_callback(
#    name = "print_callback",
#    initialize = function(message) {
#      self$message <- message
#    },
#    on_train_batch_end = function() {
#      cat("Iteration ", ctx$iter, "\n")
#    },
#    on_epoch_end = function() {
#      cat(self$message, "\n")
#    }
#  )

## ---- eval = FALSE------------------------------------------------------------
#  fitted <- net %>%
#    setup(...) %>%
#    fit(..., callbacks = list(
#      print_callback(message = "Done!")
#    ))

