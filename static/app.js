(function(){
  var img = document.getElementById('video');
  var params = new URLSearchParams(window.location.search);
  var baseParam = params.get('stream') || params.get('server');
  var base = baseParam ? String(baseParam).replace(/\/$/, '') : '';
  if (img && base) img.src = base + '/video_feed';
  var audio = new Audio(base ? base + '/alert.mp3' : 'alert.mp3');
  audio.preload = 'auto';
  audio.loop = true;
  function tick(){
    var url = base ? base + '/alert_status' : '/alert_status';
    fetch(url, {cache: 'no-store'}).then(function(r){ return r.json(); }).then(function(j){
      var a = !!(j && j.alert);
      if (a) {
        if (audio.paused) audio.play().catch(function(){});
      } else {
        if (!audio.paused) { audio.pause(); audio.currentTime = 0; }
      }
    }).catch(function(){});
    setTimeout(tick, 500);
  }
  tick();
})();