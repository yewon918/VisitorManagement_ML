<?php
$host = 'localhost';
$user = 'ferrydraw';
$pass = 'hsh0729!';
$dbName = 'ferrydraw';

//mysql connection
$connect = mysqli_connect($host, $user, $pass) or die("error");
//select db
mysql_select_db($dbName, $connect);

if($_REQUEST['select']=="show"){
    $sql = 'select * from signeduser_table';
    $result = mysql_query($sql, $connect) or die(mysql_error());

    while($array = mysql_fetch_array($result))
    {
        echo $array['id']."?";
        echo $array['pass']."&";
    }
}
?>