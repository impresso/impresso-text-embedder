
# Set the default logging level to INFO
LOGGING_LEVEL ?= INFO

# Define the logging functions
# log.debug: Prints a debug message if the debug level is set to DEBUG
# Usage: $(call log.debug, VAR_OR_STRING)
define log.debug
	$(if $(filter $(LOGGING_LEVEL),DEBUG), \
		$(if $(call variable_defined,$1), \
			$(info DEBUG:: $(strip $1) = "$($(strip $1))"), \
			$(if $(strip $1), \
				$(info DEBUG:: $(strip $1)), \
				$(info DEBUG:: "$(strip $1)") \
			) \
		) \
	)
endef

# log.info: Prints an info message if the debug level is set to INFO or DEBUG
# Usage: $(call log.info, VAR_OR_STRING)
define log.info
	$(if $(or $(filter $(LOGGING_LEVEL),INFO),$(filter $(LOGGING_LEVEL),DEBUG)), \
		$(if $(call variable_defined,$1), \
			$(info INFO:: $(strip $1) = "$($(strip $1))"), \
			$(if $(strip $1), \
				$(info INFO:: $(strip $1)), \
				$(info INFO:: "$(strip $1)") \
			) \
		) \
	)
endef

# log.warning: Prints a warning message if the debug level is set to WARNING, INFO, or DEBUG
# Usage: $(call log.warning, VAR_OR_STRING)
define log.warning
	$(if $(or $(filter $(LOGGING_LEVEL),WARNING),$(filter $(LOGGING_LEVEL),INFO),$(filter $(LOGGING_LEVEL),DEBUG)), \
		$(if $(call variable_defined,$1), \
			$(info WARNING:: $(strip $1) = "$($(strip $1))"), \
			$(if $(strip $1), \
				$(info WARNING:: $(strip $1)), \
				$(info WARNING:: "$(strip $1)") \
			) \
		) \
	)
endef

# log.error: Prints an error message if the debug level is set to ERROR, WARNING, INFO, or DEBUG
# Usage: $(call log.error, VAR_OR_STRING)
define log.error
	$(if $(or $(filter $(LOGGING_LEVEL),ERROR),$(filter $(LOGGING_LEVEL),WARNING),$(filter $(LOGGING_LEVEL),INFO),$(filter $(LOGGING_LEVEL),DEBUG)), \
		$(if $(call variable_defined,$1), \
			$(info ERROR:: $(strip $1) = "$($(strip $1))"), \
			$(if $(strip $1), \
				$(info ERROR:: $(strip $1)), \
				$(info ERROR:: "$(strip $1)") \
			) \
		) \
	)
endef

# Define a function to check if a variable is defined
define variable_defined
$(strip $(foreach v,$(1),$(if $(value $(v)),$(v))))
endef

# Example usage
FOO = bar
test_debug_level: LOGGING_LEVEL=DEBUG
test_debug_level:
	# Debug level set to DEBUG, prints debug and info messages
	$(call log.debug, FOO)
	$(call log.info, FOO)
	$(call log.warning, FOO)
	$(call log.error, FOO)
	$(call log.debug, "This is a debug message")
	$(call log.info, "This is an info message")
	$(call log.warning, "This is a warning message")
	$(call log.error, "This is an error message")

test_info_level: LOGGING_LEVEL=INFO
test_info_level:
	# Debug level set to INFO, prints info, warning, and error messages
	$(call log.debug, FOO)
	$(call log.info, FOO)
	$(call log.warning, FOO)
	$(call log.error, FOO)
	$(call log.debug, "This is a debug message")
	$(call log.info, "This is an info message")
	$(call log.warning, "This is a warning message")
	$(call log.error, "This is an error message")

test_warning_level: LOGGING_LEVEL=WARNING
test_warning_level:
	# Debug level set to WARNING, prints warning and error messages
	$(call log.debug, FOO)
	$(call log.info, FOO)
	$(call log.warning, FOO)
	$(call log.error, FOO)
	$(call log.debug, "This is a debug message")
	$(call log.info, "This is an info message")
	$(call log.warning, "This is a warning message")
	$(call log.error, "This is an error message")

test_error_level: LOGGING_LEVEL=ERROR
test_error_level:
	# Debug level set to ERROR, prints only error messages
	$(call log.debug, FOO)
	$(call log.info, FOO)
	$(call log.warning, FOO)
	$(call log.error, FOO)
	$(call log.debug, "This is a debug message")
	$(call log.info, "This is an info message")
	$(call log.warning, "This is a warning message")
	$(call log.error, "This is an error message")
