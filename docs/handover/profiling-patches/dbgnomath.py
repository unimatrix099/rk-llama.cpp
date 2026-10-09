f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
o="""    auto gelu_mul = [&](float32x4_t xv, float32x4_t g) {"""
assert s.count(o)==1
s=s.replace(o, o+"""
        static const bool nomath = std::getenv("DBG_NOMATH") != nullptr;
        if (nomath) return vmulq_f32(xv, g);""")
open(f,'w').write(s)
