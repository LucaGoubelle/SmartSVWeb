<?php

	require_once "../../src/controllers/forcastingService/csvTSManager.php";
	require_once "../../src/controllers/forcastingService/timeSerieTableManager.php";
	//require_once "../../src/controllers/forcastingService/timeSerieGraphManager.php";
	require_once "../../src/models/forcastingService/timeSerie.php";
	require_once "../../src/models/forcastingService/legend.php";
	session_start();

	$tsc = isset($_SESSION['tsc']) ? $_SESSION['tsc'] : NULL;
	$id = isset($_SESSION['id']) ? $_SESSION['id'] : NULL;
	$cmdOut = isset($_SESSION['cmdOut']) ? $_SESSION['cmdOut'] : NULL;
	$nrmF = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;
	$fcst = isset($_SESSION['fcst']) ? $_SESSION['fcst'] : NULL;

	//var_dump($nrmF);
	echo "<br>";
	//var_dump($cmdOut);
	//var_dump($tsc);
	

?>
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Smart SV prediction | série temporelle </title>
	<link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
	<script src="../../js/chartAPI/Chart.js"></script>
	<script src="../../js/graphics.js"></script>
</head>
<body>
<form action="../../src/quit.php" method="post"><input class="btn btn-danger" name="quitButton" value="Quit" type="submit"></form>

	<form>
		<table style="width: 50%;">
			<tr>
				<td>
					<input type="checkbox" id="allValueCheckbox" name="allValue">
					<label for="allValueCheckbox">Toutes les valeurs</label><br><br>
					<!--<label for="beginDate">Date début : </label>-->
					<!--<input type="date" id="beginDate" name="beginDate" value="2021-01-01" min="2018-01-01" max="2018-12-31"><br>
					<label for="endDate">Date fin : </label>-->
					<!--<input type="date" id="endDate" name="endDate" value="2021-12-31"><br>-->
				</td>
				<td>
					<a href="setParameters.php"><input type="button" name="predictionButton" value="faire prediction" style="color: white;" class="btn btn-info"></a>
				</td>
			</tr>
		</table>
	</form>
	<form action="../../src/downloadPredict.php" method="POST">
		<input type="submit" class="btn btn-success" name="downloadP" value="Télécharger la prédiction sous format CSV">
	</form>

	<div>
		<?php /*if($id!=NULL){ TimeSerieTableManager::showTimeSerieIndexTable($id); }*/?>
		<?php /*if($tsc!=NULL){ for(){ TimeSerieTableManager::showTimeSerieTable($tsc); } } */?>
	</div>
	<div>
		<?php if($fcst!=NULL){TimeSerieTableManager::showTimeSerieForcastedTable($fcst);}	?>
	</div>

	<!--<canvas id="timeSerieCanvas"></canvas>-->
	<?php
		$_SESSION['tsc'] = $tsc;
	?>
<script>
</script>
</body>
</html>