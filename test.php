<?php
	require "src/controllers/csvTSManager.php";
?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Smart SV prediction | test graph csv</title>
	<link rel="stylesheet" type="text/css" href="css/style.css">
	<script src="js/chartAPI/Chart.js"></script>
	<script src="js/graphics.js"></script>
</head>
<body>
<?php
	$tsg = CSV_TS_Manager::getTS_Graph_fromCSV("data/groupDataPP2_hour_matrix_form.csv");
	$legend_json = json_encode($tsg->getLegend()->getValues());
	$timeSerie_json = json_encode($tsg->getTimeSerie(1), JSON_NUMERIC_CHECK);
	echo "<br>"; //var_dump($tsg);
?>
<div class="chartContainer">
	<canvas id="timeSerieChart"></canvas>
</div>
<script>
	drawAGraphic("timeSerieChart",<?php echo $legend_json; ?>,<?php echo $timeSerie_json; ?>);
</script>
</body>
</html>