<?php



    session_start();
    $lstColNan = $_SESSION["lstColNan"];
    $lstColNotNan = $_SESSION["lstColNotNan"];
    $lstNN_json = json_encode($lstColNotNan);
    

?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Treat Nan Values</title>
        <link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
        <script src="../../js/pretreatParam.js"></script>
    </head>
    <body>
        <div style="text-align: center;">
            <a href="columnsParameters.php"><input style="color: white;" class="btn btn-warning" value=" << Return to columns"></a><br><br><br>
            <form action="../../src/treatNanValues.php" method="POST">
                <label for="columnName">Column name: </label>
                <select class="form-select" name="columnName" id="columnName">
                    <?php
                        for($i=0;$i<count($lstColNan);$i++)
                        {
                            echo "<option value=\"".$lstColNan[$i]."\">".$lstColNan[$i]."</option>";
                        }
                    ?>
                </select>
                <label for="columnMethodName">Method Name: </label>
                <select class="form-select" name="columnMethodName" id="columnMethodName">
                    <option value="OCB">OCB</option>
                    <option value="LOCF">LOCF</option>
                    <option value="mean">mean</option>
                </select>
                <input type="submit" class="btn btn-success" value="Treat" name="treat">
            </form>
            <br>
            <form action="../../src/treatNanValues_linearRegress.php" method="POST">
            <h1>Avec regression lineaire : </h1>
                <table style="width: 80%;">
                    <tr>
                        <td>
                            <label for="columnsNotNan">Columns not Nan</label><br>
                            <div id="columnsNotNan" >
                                <?php
                                    for($i=0;$i<count($lstColNotNan);$i++)
                                    {
                                        echo "<label for='".$lstColNotNan[$i]."' >".$lstColNotNan[$i]."</label><input type='checkbox' id='".$lstColNotNan[$i]."' name='columns[]' value='".$lstColNotNan[$i]."' onchange=\"checkboxChecking(document.getElementById('".$lstColNotNan[$i]."'));\"><br>";
                                    }
                                ?>
                            </div>
                        </td>
                        <td>
                            <button style="color: white;" class="btn btn-info" type="button" onclick="addAllMethods(<?php echo $lstNN_json;?>)">&gt;&gt;</button><br><br>
                            <button style="color: white;" class="btn btn-info" type="button" onclick="removeAllMethods(<?php echo $lstNN_json;?>)">&lt;&lt;</button>
                        </td>
                        <td>
                            <label for="columnsChosen">Columns chosen</label><br>
                            <textarea id="columnsChosen" name="columnsChosen" rows="25" cols="30"></textarea>
                        </td>
                        <td>
                            <label for="columnToTreat">Column to treat</label><br>
                            <select class="form-select" name="columnName">
                            <?php
                                for($i=0;$i<count($lstColNan);$i++){
                                    echo "<option value=\"".$lstColNan[$i]."\">".$lstColNan[$i]."</option>";
                                }
                            ?>
                            </select>
                        </td>
                        <td>
                            <input type="submit" class="btn btn-success" value="Treat" name="treatNanLR">
                        </td>
                    </tr>
                </table>
            </form>
        </div>
        <script>

        </script>
    </body>
</html>