<?php
	require "/../src/controllers/csvTSManager.php";
	session_start();

	$tsg = $_SESSION['tsg'];
	$legend_json = json_encode($tsg->getLegend()->getValues());
	$timeSerie_json = json_encode($tsg->getOriginalSerie(), JSON_NUMERIC_CHECK);
	$timeSeriePredict_json = json_encode($tsg->getForcastedSerie(), JSON_NUMERIC_CHECK);

?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Smart SV prediction | effectuer prediction</title>
	<link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
	<script src="../../js/chartAPI/Chart.js"></script>
	<script src="../../js/graphics.js"></script>
</head>
<body>

	<form>
		<table style="width: 80%;">
			<tr>
				<td>
					<input type="checkbox" name="allValueCheckbox" id="allValueCheckbox">
					<label for="allValueCheckbox">Toutes les valeurs</label><br><br>
				</td>
				<td>
					<label for="lengthPredict">Longueur</label>
					<input type="text" id="lengthPredict" name="lengthPredict"><br><br>
				</td>
			</tr>
			<tr>
				<td>
					<input type="date" id="from" name="fromDate"><br>
				</td>
				<td>
					<input type="button" name="refreshButton" class="submitButton" value="refresh">
				</td>
			</tr>
		</table>	
	</form>

	<canvas id="timeSerieAndPredictCanvas"></canvas>

<script>
	drawAGraphicWithPredict("timeSerieAndPredictCanvas",<?php echo $legend_json; ?>,<?php echo $timeSerie_json; ?>,<?php echo $timeSeriePredict_json?>);
</script>
</body>
</html>