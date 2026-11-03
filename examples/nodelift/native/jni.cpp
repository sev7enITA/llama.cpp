#include <jni.h>
#include "core.h"
extern "C" JNIEXPORT jlong JNICALL Java_dev_nodelift_companion_NativeEngine_create(JNIEnv *,jobject) {return reinterpret_cast<jlong>(nl_create());}
extern "C" JNIEXPORT jstring JNICALL Java_dev_nodelift_companion_NativeEngine_request(JNIEnv * env,jobject,jlong handle,jstring input) {
    const char * text=env->GetStringUTFChars(input,nullptr);
    if(!text)return nullptr; // JNI already has a pending allocation exception.
    char * result=nl_call(reinterpret_cast<void *>(handle),text);env->ReleaseStringUTFChars(input,text);
    if(!result)return env->NewStringUTF("{\"error\":\"native allocation failed\"}");
    jstring value=env->NewStringUTF(result);nl_free_string(result);return value;
}
extern "C" JNIEXPORT void JNICALL Java_dev_nodelift_companion_NativeEngine_interrupt(JNIEnv *,jobject,jlong handle) {nl_cancel(reinterpret_cast<void *>(handle));}
extern "C" JNIEXPORT void JNICALL Java_dev_nodelift_companion_NativeEngine_dispose(JNIEnv *,jobject,jlong handle) {nl_destroy(reinterpret_cast<void *>(handle));}
