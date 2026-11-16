// glsl_autocal_pixel.glsl - automatic calibration pattern
// uDot = (x px, y px [bottom-up], radius px, on 0/1)
// Black everywhere except one soft-edged white dot.

uniform vec4 uDot;

out vec4 fragColor;

void main() {
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = vUV.st * res;
    float d = length(px - uDot.xy);
    float a = (1.0 - smoothstep(uDot.z - 1.5, uDot.z + 1.5, d)) * uDot.w;
    fragColor = TDOutputSwizzle(vec4(vec3(a), 1.0));
}
