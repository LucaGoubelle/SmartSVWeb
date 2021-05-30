//basics function to draw the graphs
//author: Luca Goubelle

function drawAGraphic(canvasId, x, y){
	var ctx = document.getElementById(canvasId).getContext("2d");
	var chart = new Chart(ctx, {
		//type of graph
		type:"line",
		//data
		data: {
			labels: x,
			datasets:[{
				label: "time serie",
				backgroundColor: 'rgba(0,100,255,0.2)',
				borderColor: 'rgba(0,100,255,1)',
				data: y
			}]
		},
		//option
		options: {}
	});
}

function drawAGraphicWithPredict(canvasId, x, y, yPredict){
	var ctx = document.getElementById(canvasId).getContext("2d");
	var chart = new Chart(ctx, {
		//type of graph
		type:"line",
		//data
		data: {
			labels: x,
			datasets:[
				{
					label: "time serie original",
					backgroundColor: 'rgba(0,100,255,0.2)',
					borderColor: 'rgba(0,100,255,1)',
					data: y
				},
				{
					label: "time serie predicted",
					backgroundColor: 'rgba(255,0,0,0.2)',
					borderColor: 'rgba(255,0,0,1)',
					data: yPredict
				}
			]
		},
		//option
		options: {}
	});
}