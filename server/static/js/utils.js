const msg = {
    OK: 0,
    WARNING: 1,
    FAILED: 2,
    ACCEPTED: 3
}


function showAlert(type, txt) {
    $("#toastAlertTxt").text(txt);
    switch (type) {
        case msg.OK:
            $("#toastAlertTxt").css("background-color", "LightGreen");
            break;
        case msg.WARNING:
            $("#toastAlertTxt").css("background-color", "Yellow");
            break;
        case msg.FAILED:
            $("#toastAlertTxt").css("background-color", "Tomato");
            break;
        case msg.ACCEPTED:
            $("#toastAlertTxt").css("background-color", "SkyBlue");
            break;
    }
    $("#toastAlert").toast("show");
}