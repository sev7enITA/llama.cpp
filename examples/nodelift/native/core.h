#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void * nl_create(void);
// Calls on one handle serialize. Cancel requests cooperative cancellation of
// decode work; observation depends on the backend. Model loading is not
// interruptible through this API. Embedders must gate future dispatch on their
// own cancellation flag. A new call clears cancel.
char * nl_call(void * handle, const char * request_json);
void nl_cancel(void * handle);
void nl_free_string(char * value);
void nl_destroy(void * handle);
#ifdef __cplusplus
}
#endif
