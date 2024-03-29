test_that("early stopping with patience = 1", {

  fit_with_callback <- function(cb, epochs = 25) {
    model <- get_model()
    dl <- get_dl()

    suppressMessages({
      expect_message({
        mod <- model %>%
          setup(
            loss = torch::nn_mse_loss(),
            optimizer = torch::optim_adam,
          ) %>%
          set_hparams(input_size = 10, output_size = 1) %>%
          fit(dl, verbose = TRUE, epochs = epochs, callbacks = list(cb))
      })
    })
    mod
  }

  # since min_delta = 100 (large number) we expect that we will only train for
  # 2 epochs. The first one being to get a 'current best' value and the second
  # one will show no improvement thus stop training.
  mod <- fit_with_callback(luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 1,
    min_delta = 100
  ))
  expect_equal(nrow(get_metrics(mod)), 2)

  # when patience equal 2 we expect to train for at least 2 epochs.
  mod <- fit_with_callback(luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 2,
    min_delta = 100
  ))
  expect_equal(nrow(get_metrics(mod)), 3)

  # we have now scpecified that min_epochs = 5, so we must traiin for at least 5
  # epochs. However, when we are done the counter should be already updated and
  # ready to stop training.
  mod <- fit_with_callback(epochs = c(5, 25), luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 2,
    min_delta = 100
  ))
  expect_equal(nrow(get_metrics(mod)), 5)

  # if the baseline is 0, we expect to stop in the first epoch.
  mod <- fit_with_callback(epochs = c(1, 25), luz_callback_early_stopping(
    monitor = "train_loss",
    patience = 1,
    baseline = 0
  ))
  expect_equal(nrow(get_metrics(mod)), 1)

})


test_that("early stopping", {
  torch::torch_manual_seed(1)
  set.seed(1)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  expect_snapshot({
    expect_message({
      output <- mod %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = list(
          luz_callback_early_stopping(monitor = "train_loss", patience = 1,
                                      min_delta = 0.02)
        ))
    })
  })

  expect_snapshot({
    expect_message({
      output <- mod %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = list(
          luz_callback_early_stopping(monitor = "train_loss", patience = 5,
                                      baseline = 0.001)
        ))
    })
  })

  # the new callback breakpoint is used
  x <- 0
  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 25, callbacks = list(
      luz_callback_early_stopping(monitor = "train_loss", patience = 5,
                                  baseline = 0.001),
      luz_callback(on_early_stopping = function() {
        x <<- 1
      })()
    ))

  expect_equal(x, 1)

  # metric that is not the loss

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
      metrics = luz_metric_mae()
    )

  expect_snapshot({
    expect_message({
      output <- mod %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = list(
          luz_callback_early_stopping(monitor = "train_mae", patience = 2,
                                      baseline = 0.91, min_delta = 0.01)
        ))
    })
  })


})

test_that("model checkpoint callback works", {


  torch::torch_manual_seed(1)
  set.seed(1)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  tmp <- tempfile(fileext = "/")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 5, callbacks = list(
      luz_callback_model_checkpoint(path = tmp, monitor = "train_loss",
                                    save_best_only = FALSE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 5)

  tmp <- tempfile(fileext = "/")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 10, callbacks = list(
      luz_callback_model_checkpoint(path = tmp, monitor = "train_loss",
                                    save_best_only = TRUE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 10)

  torch::torch_manual_seed(2)
  set.seed(2)

  model <- get_model()
  dl <- get_dl()

  mod <- model %>%
    setup(
      loss = torch::nn_mse_loss(),
      optimizer = torch::optim_adam,
    )

  tmp <- tempfile(fileext = "/")

  output <- mod %>%
    set_hparams(input_size = 10, output_size = 1) %>%
    fit(dl, verbose = FALSE, epochs = 5, callbacks = list(
      luz_callback_model_checkpoint(path = tmp, monitor = "train_loss",
                                    save_best_only = TRUE)
    ))

  files <- fs::dir_ls(tmp)
  expect_length(files, 5)

  x <- torch_randn(10, 10)
  preds1 <- predict(output, x)

  luz_load_checkpoint(output, files[1])
  preds2 <- predict(output, x)

  luz_load_checkpoint(output, files[5])
  preds3 <- predict(output, x)

  expect_equal_to_tensor(preds1, preds3)
  expect_true(!torch_allclose(preds1, preds2))
})

test_that("early stopping + csv logger", {

  model <- get_model()
  dl <- get_dl()

  tmp <- tempfile(fileext = ".csv")

  cb <- list(
    luz_callback_early_stopping(min_delta = 100, monitor = "train_loss"),
    luz_callback_csv_logger(tmp)
  )

  suppressMessages({
    expect_message({
      mod <- model %>%
        setup(
          loss = torch::nn_mse_loss(),
          optimizer = torch::optim_adam,
        ) %>%
        set_hparams(input_size = 10, output_size = 1) %>%
        fit(dl, verbose = TRUE, epochs = 25, callbacks = cb)
    })
  })

  expect_equal(nrow(read.csv(tmp)), nrow(get_metrics(mod)))

})

test_that("use_best_model_callback", {

  module <- torch::nn_module(
    initialize = function() {
      self$w <- torch::nn_parameter(torch::torch_tensor(100))
    },
    forward = function(x) {
      torch::torch_ones_like(x)*self$w
    }
  )

  x <- torch::torch_rand(100)
  y <- torch::torch_zeros(100)

  model <- module %>%
    setup(
      loss = torch::nnf_mse_loss,
      optimizer = torch::optim_adam
    ) %>%
    set_opt_hparams(lr = 1) %>%
    fit(list(x, y), verbose = FALSE, callbacks = list(
      luz_callback_keep_best_model("train_loss", mode = "max")
    ))

  expect_true(model$model$w$item() > 90)
})
