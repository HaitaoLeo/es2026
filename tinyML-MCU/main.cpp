#include <Arduino.h>

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>

#include <ArduTFLite.h>
#include "model.h"     // model_tflite[] / model_tflite_len  —— 由 Notebook 的 convert_to_c_array 导出
#include "scaler.h"    // INPUT_MEAN / INPUT_SCALE          —— 与 model.h 必须同源、配套更新

// ============================================================
// 1. 板级差异集中在这里
//    ESP32：ADC1 GPIO34/35；TFT 走默认 VSPI（SCK18 / MOSI23 / MISO19）
//    Pico ：GP26/GP27 为 ADC；SPI0（GP18 SCK / GP19 MOSI / GP16 MISO）
// ============================================================
#if defined(ARDUINO_ARCH_ESP32)
  // 两个电位器的引脚对调了：上面的 pot2 走 GPIO34（y 湿度轴），下面的 pot1 走 GPIO35（x 温度轴）。
  // 这样上面那根线接上面的 34、下面那根接下面的 35，两根线平行不交叉。
  const int pin_adc_x = 35;    // 下方电位器 pot1 -> 特征 x（温度：左舒适 / 右热）
  const int pin_adc_y = 34;    // 上方电位器 pot2 -> 特征 y（湿度：高潮湿 / 低干燥）
  const int pin_led   = 2;     // 板载 LED
  const int tft_cs    = 5;
  const int tft_dc    = 21;
  const int tft_rst   = 4;
  const int tft_sck   = 18;
  const int tft_mosi  = 23;
  const int tft_miso  = 19;    // 只有需要回读数据才用得上，接上不亏
  #define LOG Serial
#else
  const int pin_adc_x = 26;
  const int pin_adc_y = 27;
  const int pin_led   = 22;
  const int tft_cs    = 17;
  const int tft_dc    = 20;
  const int tft_rst   = 21;
  const int tft_sck   = 18;
  const int tft_mosi  = 19;
  const int tft_miso  = 16;
  #define LOG Serial1
#endif

// 只用 (cs, dc, rst) 三个参数 → 走硬件 SPI。
// 注意：千万别用 (cs,dc,mosi,sck,rst,miso) 那个 6 参数重载，
// 它会退化成 TFT_SOFT_SPI（GPIO 位翻转模拟 SPI），刷一屏要好几秒。
// 具体用哪三个 SPI 引脚由 setup() 里的 SPI.begin(...) 决定。
Adafruit_ILI9341 tft(tft_cs, tft_dc, tft_rst);

// ============================================================
// 2. 张量竞技场（Tensor Arena）
// ============================================================
constexpr int tensorArenaSize = 16 * 1024;
alignas(16) byte tensorArena[tensorArenaSize];

// ============================================================
// 3. 归一化参数：必须与训练时的 StandardScaler 完全一致
//    x' = (raw - mean) / scale
//    Notebook 样本是 0~1023 的坐标，所以 ADC 也必须读 0~1023（见 setup 里的 analogReadResolution）
// ============================================================
const float mean[2]  = INPUT_MEAN;
const float scale[2] = INPUT_SCALE;

// ============================================================
// 4. 类别语义（Notebook 里的字母 → 实际含义）
//    a = 潮湿舒适   b = 潮湿闷热   c = 干燥舒适   d = 干燥偏热
//    屏上用英文显示；串口仍输出字母，方便 CSV 解析
// ============================================================
constexpr int NUM_CLASSES = 4;
const char CLASS_ID[NUM_CLASSES] = {'a', 'b', 'c', 'd'};
const char* const CLASS_NAME[NUM_CLASSES] = {
  "WET COMFORT",   // a 潮湿舒适
  "WET HOT",       // b 潮湿闷热
  "DRY COMFORT",   // c 干燥舒适
  "DRY HOT"        // d 干燥偏热
};
const uint16_t CLASS_COLOR[NUM_CLASSES] = {
  0x07E0,   // a 绿
  0xF800,   // b 红
  0x07FF,   // c 青
  0xFD20    // d 橙
};

// ============================================================
// 5. 屏幕布局（ILI9341 320x240 横屏）
//    TFT_ROTATION 与 diagram.json 里屏幕元件的 rotate 是叠加关系（不是互补）：
//      diagram.json "rotate": 90（面板在画布上转 90°，排针朝左对着 ESP32）
//      + 代码 setRotation(1)（画面向同方向再转 90°）
//      两者相抵 → 文字正立，面板呈 320x240 横向
//    如果文字方向又不对：把这里在 1 / 3 之间切换（3 会得到 180°，即上下颠倒）。
// ============================================================
const uint8_t TFT_ROTATION = 1;   // 横屏：元件 rotate:90 与 setRotation 相减，90-90=0（屏右上角显示 r1）
const int SCR_W = 320;
const uint16_t C_BG    = 0x0000;   // 黑底
const uint16_t C_TEXT  = 0xFFFF;   // 白
const uint16_t C_DIM   = 0x8410;   // 灰
const uint16_t C_FRAME = 0x2945;   // 进度条/分隔线

const int ROW_Y0 = 72;             // 第一行的 y
const int ROW_H  = 26;
const int BAR_X  = 34;
const int BAR_W  = 216;
const int BAR_H  = 14;
const int PCT_X  = 256;

// ------------------------------------------------------------
// 四分类的判据是 argmax，不再是二分类时代的 result > 0.5
// ------------------------------------------------------------
int argmax(const float* values, int count) {
  int best = 0;
  for (int i = 1; i < count; i++) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

// ------------------------------------------------------------
// LED：a 最暗 … d 最亮
// ESP32 core 2.x 没有 analogWrite，要用 ledc；core 3.x 才有 analogWrite
// ------------------------------------------------------------
void ledLevel(int level) {
  const int duty = (level + 1) * 63;   // 63 / 126 / 189 / 252
#if defined(ARDUINO_ARCH_ESP32)
  #if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    analogWrite(pin_led, duty);
  #else
    ledcWrite(0, duty);
  #endif
#else
  analogWrite(pin_led, duty);
#endif
}

void ledBegin() {
#if defined(ARDUINO_ARCH_ESP32)
  ledcSetup(0, 5000, 8);
  ledcAttachPin(pin_led, 0);
#else
  pinMode(pin_led, OUTPUT);
#endif
  ledLevel(0);
}

// ------------------------------------------------------------
// 静态界面：标题、分隔线、每行的类别字母、底部图例
// 这些内容一直不变，只在 setup 里画一次，避免每帧重绘闪烁
// ------------------------------------------------------------
void screenDrawStatic() {
  tft.fillScreen(C_BG);
  tft.setTextWrap(false);

  tft.setTextSize(2);
  tft.setTextColor(C_TEXT);
  tft.setCursor(8, 6);
  tft.print("TFLite 4-Class");

  // 右上角显示当前生效的旋转值：改完 TFT_ROTATION 重新编译后，
  // 这里应同步变成 r0/r1/r2/r3 —— 如果没变，说明 Wokwi 还在跑旧固件。
  tft.setTextColor(C_DIM);
  tft.setCursor(280, 6);
  tft.setTextSize(1);
  tft.print("r");
  tft.print(TFT_ROTATION);

  tft.drawFastHLine(0, 26, SCR_W, C_FRAME);

  // 四行的类别字母（固定不变）
  tft.setTextSize(3);
  for (int i = 0; i < NUM_CLASSES; i++) {
    tft.setTextColor(CLASS_COLOR[i]);
    tft.setCursor(8, ROW_Y0 + i * ROW_H - 4);
    tft.print(CLASS_ID[i]);
  }

  tft.drawFastHLine(0, 182, SCR_W, C_FRAME);

  // 图例：把字母和含义的对应关系钉在屏幕底部
  tft.setTextSize(1);
  tft.setTextColor(C_DIM);
  tft.setCursor(8, 222);
  tft.print("a wet-comfort  b wet-hot  c dry-comfort  d dry-hot");
}

// 当前类别（大字居中）；只在类别切换时才重绘，避免无谓闪烁
void screenDrawResult(int cls, int last_cls) {
  if (cls == last_cls) {
    return;
  }
  const char* name = CLASS_NAME[cls];
  const int text_w = (int)strlen(name) * 18;      // size 3 每字符 18px
  const int x = (SCR_W - text_w) / 2;

  tft.fillRect(0, 30, SCR_W, 32, C_BG);           // 先清掉上一次的字
  tft.setTextSize(3);
  tft.setTextColor(CLASS_COLOR[cls]);
  tft.setCursor(x, 32);
  tft.print(name);
}

// 四类概率条
void screenDrawBars(const float* probs) {
  tft.setTextSize(2);
  for (int i = 0; i < NUM_CLASSES; i++) {
    const int y = ROW_Y0 + i * ROW_H;
    const int bar_y = y + 4;

    tft.fillRect(BAR_X, bar_y, BAR_W, BAR_H, C_BG);       // 清底
    tft.drawRect(BAR_X, bar_y, BAR_W, BAR_H, C_FRAME);    // 边框

    const int w = (int)(probs[i] * (BAR_W - 2) + 0.5f);
    if (w > 0) {
      tft.fillRect(BAR_X + 1, bar_y + 1, w, BAR_H - 2, CLASS_COLOR[i]);
    }

    // 百分比：位数会变，先清一块再写
    tft.fillRect(PCT_X, y - 2, 60, 20, C_BG);
    tft.setTextColor(CLASS_COLOR[i]);
    tft.setCursor(PCT_X, y);
    tft.print((int)(probs[i] * 100.0f + 0.5f));
    tft.print("%");
  }
}

// 底部：原始 ADC / 归一化后的输入 / 单次推理耗时
void screenDrawFooter(int v1, int v2, float nx, float ny, uint32_t infer_us) {
  tft.fillRect(0, 186, SCR_W, 34, C_BG);
  tft.setTextSize(1);
  tft.setTextColor(C_DIM);

  tft.setCursor(8, 188);
  tft.print("raw  x=");  tft.print(v1);
  tft.print("  y=");     tft.print(v2);

  tft.setCursor(8, 202);
  tft.print("norm x=");  tft.print(nx, 2);
  tft.print("  y=");     tft.print(ny, 2);

  tft.setCursor(214, 202);
  tft.print("infer ");
  tft.print(infer_us / 1000.0f, 2);
  tft.print(" ms");
}

void setup() {
  LOG.begin(115200);
  delay(300);                       // ESP32 上电会打印 ROM 日志，等一下再输出
  LOG.println("Hello, ESP32 + TFLite Micro!");

  // 与训练数据 0~1023 对齐：ESP32 默认 12bit(0~4095)，必须显式改成 10bit
  analogReadResolution(10);

  ledBegin();

  // ---- 屏幕 ----
#if defined(ARDUINO_ARCH_ESP32)
  // 显式指定硬件 SPI 引脚：SCK, MISO, MOSI, SS
  SPI.begin(tft_sck, tft_miso, tft_mosi, tft_cs);
#endif
  tft.begin();
  tft.setRotation(TFT_ROTATION);    // 横屏（与 diagram.json 的 rotate 配套，见文件上方说明）
  screenDrawStatic();
  LOG.println("TFT ready.");

  // ---- 模型初始化 ----
  if (!modelInit(model_tflite, tensorArena, tensorArenaSize)) {
    LOG.println("Model initialization failed!");
    tft.setTextSize(2);
    tft.setTextColor(0xF800);
    tft.setCursor(10, 110);
    tft.print("MODEL INIT FAILED");
    while (true) {
      delay(1000);
    }
  }
  LOG.println("Model initialization done.");

  // ---- 自检：确认加载的确实是四分类模型 ----
  if (modelGetOutput(NUM_CLASSES - 1) < 0.0f) {
    LOG.println("ERROR: 输出张量不足 4 维，请确认 src/model.h 是四分类版本！");
    tft.setTextSize(2);
    tft.setTextColor(0xF800);
    tft.setCursor(10, 110);
    tft.print("MODEL IS NOT 4-CLASS");
    while (true) {
      delay(1000);
    }
  }
  LOG.println("Output = 4 classes (a/b/c/d)");
  LOG.println("x,y,nx,ny,p_a,p_b,p_c,p_d,cls,label");
}

int last_cls = -1;   // 上一帧的类别，用来判断标题要不要重绘

void loop() {
  // ---- 1. 采样 ----
  const int v1 = analogRead(pin_adc_x);
  const int v2 = analogRead(pin_adc_y);

  // ---- 2. 归一化（与训练同源参数） ----
  const float nx = (v1 - mean[0]) / scale[0];
  const float ny = (v2 - mean[1]) / scale[1];

  // ---- 3. 填输入张量（shape = [1, 2]） ----
  if (!modelSetInput(nx, 0) || !modelSetInput(ny, 1)) {
    LOG.println("SetInput failed!");
    delay(500);
    return;
  }

  // ---- 4. 推理（顺带量一下耗时，屏上显示） ----
  const uint32_t t0 = micros();
  const bool ok = modelRunInference();
  const uint32_t infer_us = micros() - t0;
  if (!ok) {
    LOG.println("RunInference failed!");
    delay(500);
    return;
  }

  // ---- 5. 取 4 个概率并 argmax ----
  float probs[NUM_CLASSES];
  for (int i = 0; i < NUM_CLASSES; i++) {
    probs[i] = modelGetOutput(i);
  }
  const int cls = argmax(probs, NUM_CLASSES);

  // ---- 6. 串口 CSV（字母 + 英文含义） ----
  LOG.print(v1);                        LOG.print(',');
  LOG.print(v2);                        LOG.print(',');
  LOG.print(nx, 3);                     LOG.print(',');
  LOG.print(ny, 3);                     LOG.print(',');
  for (int i = 0; i < NUM_CLASSES; i++) {
    LOG.print(probs[i], 3);
    LOG.print(',');
  }
  LOG.print(CLASS_ID[cls]);             LOG.print(',');
  LOG.println(CLASS_NAME[cls]);

  // ---- 7. 屏幕 ----
  screenDrawResult(cls, last_cls);
  last_cls = cls;
  screenDrawBars(probs);
  screenDrawFooter(v1, v2, nx, ny, infer_us);

  // ---- 8. LED：a 最暗 … d 最亮 ----
  ledLevel(cls);

  delay(200);
}
