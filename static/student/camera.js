
/*
 * Hàm dùng chung cho trang đăng ký và trang thi:
 * - Mở camera
 * - Chụp khung hình
 * - Gửi dữ liệu JSON tới Flask API
 * - Lặp gửi ảnh nhưng không tạo request chồng nhau
 */

/* ---------- 1. Mở camera ---------- */

async function startCamera(video) {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    throw new Error("INSECURE");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user"
    },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((resolve, reject) => {
    if (video.readyState >= 1 && video.videoWidth > 0) {
      resolve();
      return;
    }

    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("VIDEO_ERROR"));
  });

  await video.play();

  return stream;
}


/* ---------- 2. Thông báo lỗi camera ---------- */

function cameraErrorMessage(e) {
  if (e && e.message === "INSECURE") {
    return "Trình duyệt chỉ cho mở camera khi truy cập bằng localhost hoặc HTTPS. Hãy mở trang bằng địa chỉ https:// hoặc http://localhost.";
  }

  if (e && e.name === "NotAllowedError") {
    return "Bạn chưa cho phép dùng camera. Bấm biểu tượng ổ khóa cạnh thanh địa chỉ, chọn cho phép Camera rồi tải lại trang.";
  }

  if (e && e.name === "NotFoundError") {
    return "Không tìm thấy camera. Hãy kiểm tra webcam đã cắm chưa.";
  }

  if (e && e.name === "NotReadableError") {
    return "Camera đang được ứng dụng khác sử dụng. Hãy tắt ứng dụng đó rồi tải lại trang.";
  }

  if (e && e.message === "VIDEO_ERROR") {
    return "Không thể khởi tạo video từ camera. Hãy thử tải lại trang.";
  }

  return "Không mở được camera: " +
    (e && e.message ? e.message : e);
}


/* ---------- 3. Chụp khung hình ---------- */

function grabFrame(video, canvas, quality) {
  if (
    !video ||
    !canvas ||
    !video.videoWidth ||
    !video.videoHeight
  ) {
    throw new Error("Camera chưa sẵn sàng để chụp ảnh.");
  }

  const width = 640;
  const height = Math.round(
    width * video.videoHeight / video.videoWidth
  );

  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");

  if (!context) {
    throw new Error("Không thể tạo canvas context.");
  }

  context.drawImage(video, 0, 0, width, height);

  return canvas.toDataURL(
    "image/jpeg",
    quality ?? 0.7
  );
}


/* ---------- 4. Gửi JSON tới Flask API ---------- */

async function postJSON(url, body, timeout = 10000) {
  const controller = new AbortController();

  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeout);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body),
      signal: controller.signal
    });

    let data;

    try {
      data = await response.json();
    } catch (e) {
      data = {
        ok: false,
        message: "Máy chủ trả về dữ liệu không hợp lệ."
      };
    }

    data._http = response.status;

    return data;

  } catch (e) {
    if (e.name === "AbortError") {
      return {
        ok: false,
        _http: 0,
        message: "Yêu cầu tới máy chủ đã hết thời gian chờ."
      };
    }

    console.error("API error:", e);

    return {
      ok: false,
      _http: 0,
      message: "Mất kết nối tới máy chủ."
    };

  } finally {
    clearTimeout(timeoutId);
  }
}


/* ---------- 5. Vòng lặp gửi khung hình ---------- */

/*
 * Chờ fn() hoàn tất rồi mới chạy lần tiếp theo.
 * Không tạo request chồng nhau.
 */

function frameLoop(fn, interval) {
  let stopped = false;
  let timer = null;

  async function tick() {
    if (stopped) return;

    const startTime = Date.now();

    try {
      await fn();
    } catch (e) {
      console.error("Frame loop error:", e);
    }

    if (!stopped) {
      const elapsed = Date.now() - startTime;
      const delay = Math.max(50, interval - elapsed);

      timer = setTimeout(tick, delay);
    }
  }

  tick();

  return function stop() {
    stopped = true;

    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };
}


/* ---------- 6. Tắt camera ---------- */

function stopStream(stream) {
  if (!stream) return;

  stream.getTracks().forEach(function (track) {
    track.stop();
  });
}
