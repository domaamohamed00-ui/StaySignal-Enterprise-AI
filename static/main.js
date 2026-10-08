async function submitBooking() {
    const payload = {
        hotel: document.getElementById('hotel').value,
        lead_time: parseInt(document.getElementById('lead_time').value) || 0,
        arrival_date_year: 2026,
        arrival_date_month: "October",
        arrival_date_week_number: 41,
        arrival_date_day_of_month: 8,
        stays_in_weekend_nights: parseInt(document.getElementById('stays_in_weekend_nights').value) || 0,
        stays_in_week_nights: parseInt(document.getElementById('stays_in_week_nights').value) || 0,
        adults: parseInt(document.getElementById('adults').value) || 1,
        children: parseFloat(document.getElementById('children').value) || 0.0,
        babies: 0,
        meal: "BB",
        country: "PRT",
        market_segment: "Online TA",
        distribution_channel: "TA/TO",
        is_repeated_guest: 0,
        previous_cancellations: parseInt(document.getElementById('previous_cancellations').value) || 0,
        previous_bookings_not_canceled: 0,
        reserved_room_type: "A",
        assigned_room_type: "A",
        booking_changes: 0,
        deposit_type: document.getElementById('deposit_type').value,
        agent: 9.0,
        company: 0.0,
        days_in_waiting_list: 0,
        customer_type: document.getElementById('customer_type').value,
        adr: parseFloat(document.getElementById('adr').value) || 0.0,
        required_car_parking_spaces: parseInt(document.getElementById('required_car_parking_spaces').value) || 0,
        total_of_special_requests: parseInt(document.getElementById('total_of_special_requests').value) || 0
    };

    const scoreEl = document.getElementById('gaugeScore');
    const titleEl = document.getElementById('gaugeTitle');
    const subEl = document.getElementById('gaugeSub');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === 'success') {
            const probPct = (data.cancellation_probability * 100).toFixed(1);
            scoreEl.innerText = `${probPct}%`;

            if (data.is_canceled_prediction === 1) {
                scoreEl.className = "display-1 fw-bold text-danger";
                titleEl.innerText = "High Cancellation Risk";
                titleEl.className = "fw-bold text-danger";
                subEl.innerText = "This booking has a high probability of cancellation.";
            } else {
                scoreEl.className = "display-1 fw-bold text-success";
                titleEl.innerText = "Safe & Confirmed Booking";
                titleEl.className = "fw-bold text-success";
                subEl.innerText = "Low risk detected. Guest is likely to stay.";
            }
        }
    } catch (e) {
        alert("API Error! Make sure FastAPI server is running.");
    }
}

function loadExample(type) {
    if (type === 'safe') {
        document.getElementById('hotel').value = 'City Hotel';
        document.getElementById('lead_time').value = '10';
        document.getElementById('deposit_type').value = 'No Deposit';
        document.getElementById('stays_in_week_nights').value = '2';
        document.getElementById('stays_in_weekend_nights').value = '1';
        document.getElementById('adults').value = '2';
        document.getElementById('children').value = '1';
        document.getElementById('adr').value = '100';
        document.getElementById('customer_type').value = 'Transient';
        document.getElementById('previous_cancellations').value = '0';
        document.getElementById('required_car_parking_spaces').value = '1';
        document.getElementById('total_of_special_requests').value = '2';
    } else if (type === 'risky') {
        document.getElementById('hotel').value = 'City Hotel';
        document.getElementById('lead_time').value = '450';
        document.getElementById('deposit_type').value = 'Non Refund';
        document.getElementById('stays_in_week_nights').value = '4';
        document.getElementById('stays_in_weekend_nights').value = '2';
        document.getElementById('adults').value = '1';
        document.getElementById('children').value = '0';
        document.getElementById('adr').value = '250';
        document.getElementById('customer_type').value = 'Transient';
        document.getElementById('previous_cancellations').value = '3';
        document.getElementById('required_car_parking_spaces').value = '0';
        document.getElementById('total_of_special_requests').value = '0';
    }
    submitBooking();
}

function resetForm() {
    document.getElementById('bookingForm').reset();
    document.getElementById('gaugeScore').innerText = "-";
    document.getElementById('gaugeScore').className = "display-1 fw-bold text-primary";
    document.getElementById('gaugeTitle').innerText = "Waiting for details";
    document.getElementById('gaugeTitle').className = "fw-bold text-light";
    document.getElementById('gaugeSub').innerText = "Fill in the booking and the score appears here.";
}