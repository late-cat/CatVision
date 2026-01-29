(function () {
  var video = document.getElementById('video');
  var params = new URLSearchParams(window.location.search);
  var baseParam = params.get('stream') || params.get('server');
  var base = baseParam ? String(baseParam).replace(/\/$/, '') : '';

  if (video && base) {
    video.src = base + '/video_feed';
  }

  var audio = new Audio(base ? base + '/alert.mp3' : '/alert.mp3');
  audio.preload = 'auto';
  audio.loop = true;

  function checkAlert() {
    var url = base ? base + '/alert_status' : '/alert_status';

    fetch(url, { cache: 'no-store' })
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        var isAlert = !!(data && data.alert);

        if (isAlert && audio.paused) {
          audio.play().catch(function () { });
        } else if (!isAlert && !audio.paused) {
          audio.pause();
          audio.currentTime = 0;
        }
      })
      .catch(function () { });

    setTimeout(checkAlert, 500);
  }

  checkAlert();
})();